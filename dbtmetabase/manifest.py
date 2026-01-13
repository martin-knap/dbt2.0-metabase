from __future__ import annotations

import dataclasses as dc
import json
import logging
import re
from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from .format import NullValue

_logger = logging.getLogger(__name__)

# Namespace for meta fields, e.g. metabase.field
_META_NS = "metabase"
# Allowed namespace fields
_COMMON_META_FIELDS = [
    "display_name",
    "visibility_type",
    "description",
]
# Must be covered by Column attributes
_COLUMN_META_FIELDS = _COMMON_META_FIELDS + [
    "semantic_type",
    "has_field_values",
    "coercion_strate`gy",
    "number_style",
    "decimals",
    "currency",
]
# Must be covered by Model attributes
_MODEL_META_FIELDS = _COMMON_META_FIELDS + [
    "points_of_interest",
    "caveats",
]

# Default values for non-standard sources
DEFAULT_DATABASE = ""
DEFAULT_SCHEMA = "PUBLIC"

# Foreign key constraint: "schema.model (column)" / "model (column)"
_CONSTRAINT_FK_PARSER = re.compile(r"(?P<model>.+)\s+\((?P<column>.+)\)")


class Manifest:
    """dbt manifest reader."""

    def __init__(self, path: str | Path):
        """Reader for compiled dbt manifest.json file.

        Args:
            path (Union[str, Path]): Path to dbt manifest.json (usually under target/).
        """

        self.path = Path(path).expanduser()

    def read_models(self) -> Sequence[Model]:
        """Reads dbt models in Metabase-friendly format.

        Returns:
            Sequence[Model]: List of dbt models in Metabase-friendly format.
        """

        with open(self.path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        models: MutableSequence[Model] = []

        for node in manifest["nodes"].values():
            if node["resource_type"] != "model":
                continue

            name = node["name"]
            if node["config"]["materialized"] == "ephemeral":
                _logger.debug("Skipping ephemeral model '%s'", name)
                continue

            models.append(self._read_model(manifest, node, Group.nodes))

        for node in manifest["sources"].values():
            if node["resource_type"] != "source":
                continue

            models.append(
                self._read_model(manifest, node, Group.sources, node["source_name"])
            )

        return models

    def _read_model(
        self,
        manifest: Mapping,
        manifest_model: Mapping,
        group: Group,
        source: str | None = None,
    ) -> Model:
        database = manifest_model["database"]
        schema = manifest_model["schema"]
        unique_id = manifest_model["unique_id"]

        relationships = self._read_relationships(manifest, group, unique_id)

        columns = [
            self._read_column(column, schema, relationships.get(column["name"]))
            for column in manifest_model.get("columns", {}).values()
        ]

        meta = self._scan_fields(
            manifest_model.get("meta", {}),
            fields=_MODEL_META_FIELDS,
            ns=_META_NS,
        )
        description = meta.pop("description", manifest_model.get("description"))

        return Model(
            database=database,
            schema=schema,
            group=group,
            name=manifest_model["name"],
            alias=manifest_model.get(
                "alias", manifest_model.get("identifier", manifest_model["name"])
            ),
            description=description,
            columns=columns,
            unique_id=unique_id,
            source=source,
            tags=manifest_model.get("tags", []),
            **meta,
        )

    def _read_column(
        self,
        manifest_column: Mapping,
        schema: str,
        relationship: Mapping | None,
    ) -> Column:
        meta = self._scan_fields(
            manifest_column.get("meta", {}),
            fields=_COLUMN_META_FIELDS,
            ns=_META_NS,
        )
        description = meta.pop("description", manifest_column.get("description"))

        column = Column(
            name=manifest_column.get("name", ""),
            description=description,
            **meta,
        )

        # DEBUG: Log constraints
        col_name = manifest_column.get("name", "")
        constraints = manifest_column.get("constraints", [])
        if constraints:
            _logger.debug(f"Column {col_name} has constraints: {constraints}")

        self._set_column_relationship(
            manifest_column=manifest_column,
            column=column,
            schema=schema,
            relationship=relationship,
        )

        return column

    def _read_relationships(
        self,
        manifest: Mapping,
        group: Group,
        unique_id: str,
    ) -> Mapping[str, Mapping[str, str]]:
        relationships = {}

        # Check if model has any children (tests) in child_map
        if unique_id not in manifest["child_map"]:
            return relationships

        for child_id in manifest["child_map"][unique_id]:
            child = manifest.get("nodes", {}).get(child_id, {})
            child_name = child.get("alias", child.get("name"))

            if (
                child.get("resource_type") == "test"
                and child.get("test_metadata", {}).get("name") == "relationships"
            ):
                # To get the name of the foreign table, we could use child[test_metadata][kwargs][to], which
                # would return the ref() written in the test, but if the model has an alias, that's not enough.
                # Using child[depends_on][nodes] and excluding the current model is better.

                # Nodes contain at most two tables: referenced model and current model (optional).
                depends_on_nodes = list(child["depends_on"][group])

                # Relationships on disabled models mention them in refs but not depends_on,
                # which confuses the filtering logic that follows.
                depends_on_names = {n.split(".")[-1] for n in depends_on_nodes}
                mismatched_refs = []
                for ref in child["refs"]:
                    ref_name = ""
                    if isinstance(ref, dict):  # current manifest
                        ref_name = ref["name"]
                    elif isinstance(ref, list):  # old manifest
                        ref_name = ref[0]
                    if ref_name not in depends_on_names:
                        mismatched_refs.append(ref_name)

                if mismatched_refs:
                    _logger.debug(
                        "Mismatched refs %s with depends_on for relationship '%s', skipping",
                        mismatched_refs,
                        child_name,
                    )
                    continue

                # dbt 2.0: depends_on_nodes contains [source_model, target_model]
                # We want to process relationships where source_model is the current model
                # and skip incoming relationships where target_model is the current model

                if len(depends_on_nodes) > 2:
                    _logger.warning(
                        "Unexpected %d depends_on for relationship '%s' instead of <=2, skipping",
                        len(depends_on_nodes),
                        child_name,
                    )
                    continue

                if len(depends_on_nodes) != 2:
                    _logger.warning(
                        "Expected 2 dependencies for relationship '%s', got %d, skipping",
                        child_name,
                        len(depends_on_nodes),
                    )
                    continue

                # Identify source (model with FK column) and target (model with PK)
                # The test is defined on the source model, so first node should be source
                source_model_id = depends_on_nodes[0]
                target_model_id = depends_on_nodes[1]

                # Skip if this is an incoming relationship (we're the target, not the source)
                if target_model_id == unique_id:
                    _logger.debug(
                        "Skipping incoming relationship test '%s' - we are the FK target, not source",
                        child_name,
                    )
                    continue

                # Process only if we are the source model
                if source_model_id != unique_id:
                    _logger.debug(
                        "Skipping relationship test '%s' - not defined on current model",
                        child_name,
                    )
                    continue

                # Now target_model_id is the FK target
                depends_on_id = target_model_id
                depends_on_group = Group.from_unique_id(depends_on_id)
                if not depends_on_group:
                    _logger.debug("Unknown group dependency '%s'", depends_on_id)
                    continue

                fk_target_model = manifest[depends_on_group].get(depends_on_id, {})
                fk_target_table = (
                    fk_target_model.get("alias")
                    or fk_target_model.get("identifier")
                    or fk_target_model.get("name")
                )
                if not fk_target_table:
                    _logger.debug("Cannot resolve dependency for '%s'", depends_on_id)
                    continue

                fk_target_schema = fk_target_model.get("schema", DEFAULT_SCHEMA)
                fk_target_table = f"{fk_target_schema}.{fk_target_table}"

                # dbt 2.0: Parse generated_sql_file to extract field parameter
                test_kwargs = child["test_metadata"]["kwargs"]
                fk_target_field = None

                if "generated_sql_file" in child:
                    # dbt 2.0 stores the rendered test call in generated_sql_file
                    # Example: {{ test_relationships(column_name="product_id", field="product_id", model=..., to=ref('dim_product')) }}
                    # The path is relative to project root, but manifest is in target/
                    # So we need to go up one level from manifest location
                    sql_file_path = child["generated_sql_file"]
                    manifest_path = Path(self.path)
                    # manifest is at dbt/target/manifest.json
                    # sql_file_path is target/generic_tests/...sql
                    # We need dbt/target/generic_tests/...sql
                    project_root = manifest_path.parent.parent  # Go from target/ to dbt/
                    sql_full_path = project_root / sql_file_path

                    try:
                        with open(sql_full_path, "r", encoding="utf-8") as f:
                            sql_content = f.read()

                        # Extract field= parameter from the test call
                        import re
                        field_match = re.search(r'field=["\']?(\w+)["\']?', sql_content)
                        if field_match:
                            fk_target_field = field_match.group(1)
                        else:
                            _logger.warning(
                                "Cannot extract target field from generated SQL for relationship '%s'",
                                child_name,
                            )
                            continue
                    except Exception as e:
                        _logger.warning(
                            "Failed to read generated SQL file for relationship '%s': %s",
                            child_name,
                            e,
                        )
                        continue
                else:
                    _logger.warning(
                        "Missing generated_sql_file for relationship '%s'",
                        child_name,
                    )
                    continue

                # Get source column name from kwargs (both dbt 1.x and 2.0)
                source_column = test_kwargs.get("column_name")
                if not source_column:
                    _logger.warning(
                        "Missing source column for relationship '%s'",
                        child_name,
                    )
                    continue

                relationships[source_column] = {
                    "fk_target_table": fk_target_table,
                    "fk_target_field": fk_target_field,
                }

        return relationships

    def _set_column_relationship(
        self,
        manifest_column: Mapping,
        column: Column,
        schema: str,
        relationship: Mapping | None,
    ):
        """Sets primary key and foreign key target on a column from constraints, meta fields or provided test relationship."""

        fk_target_table = ""
        fk_target_field = ""

        # Precedence 1: Relationship test
        if relationship:
            fk_target_table = relationship["fk_target_table"]
            fk_target_field = relationship["fk_target_field"]

        # Precedence 2: Constraints
        for constraint in manifest_column.get("constraints", []):
            if constraint["type"] == "primary_key":
                if not column.semantic_type:
                    column.semantic_type = "type/PK"

            elif constraint["type"] == "foreign_key":
                # Constraint: expression
                if constraint_expr := constraint.get("expression"):
                    constraint_fk = _CONSTRAINT_FK_PARSER.search(constraint_expr)
                    if constraint_fk:
                        fk_target_table = constraint_fk.group("model")
                        fk_target_field = constraint_fk.group("column")
                    else:
                        _logger.warning(
                            "Unparsable '%s' foreign key constraint: %s",
                            column.name,
                            constraint_expr,
                        )

                # Constraint: to + to_columns
                # Foreign key constraint `to` compiles down to "project"."schema"."model"
                elif constraint_to := constraint.get("to"):
                    constraint_to_path = constraint_to.split(".")
                    constraint_to_columns = constraint.get("to_columns", [])
                    if len(constraint_to_path) <= 3 and len(constraint_to_columns) == 1:
                        # "project"."schema"."model" -> "schema"."model"
                        fk_target_table = ".".join(constraint_to_path[-2:])
                        fk_target_field = constraint_to_columns[0]
                    else:
                        _logger.warning(
                            "Unparsable '%s' foreign key constraint: %s, %s",
                            column.name,
                            constraint_to,
                            constraint_to_columns,
                        )

        # Precedence 3: Meta fields
        meta = manifest_column.get("meta", {})
        fk_target_table = meta.get(f"{_META_NS}.fk_target_table", fk_target_table)
        fk_target_field = meta.get(f"{_META_NS}.fk_target_field", fk_target_field)

        if not fk_target_table or not fk_target_field:
            if fk_target_table or fk_target_table:
                _logger.warning(
                    "Foreign key requires table and field for column '%s'",
                    column.name,
                )
            return

        fk_target_table_path = fk_target_table.split(".")
        if len(fk_target_table_path) == 1 and schema:
            fk_target_table_path.insert(0, schema)

        column.semantic_type = "type/FK"
        column.fk_target_table = ".".join([x.strip('"') for x in fk_target_table_path])
        column.fk_target_field = fk_target_field.strip('"')
        _logger.debug(
            "Relation from '%s' to '%s.%s'",
            column.name,
            column.fk_target_table,
            column.fk_target_field,
        )

    @staticmethod
    def _scan_fields(
        t: Mapping, fields: Iterable[str], ns: str
    ) -> MutableMapping[str, Any]:
        """Reads meta fields from a schem object.

        Args:
            t (Mapping): Target to scan for fields.
            fields (Iterable): List of fields to accept.
            ns (str): Field namespace (separated by .).

        Returns:
            Mapping: Field values.
        """

        vals = {}
        for field in fields:
            if f"{ns}.{field}" in t:
                value = t[f"{ns}.{field}"]
                vals[field] = value if value is not None else NullValue
        return vals


class Group(str, Enum):
    nodes = "nodes"
    sources = "sources"

    @staticmethod
    def from_unique_id(unique_id: str) -> Group | None:
        prefix = unique_id.split(".")[0]
        if prefix == "source":
            return Group.sources
        elif prefix == "model":
            return Group.nodes
        return None


@dc.dataclass
class Column:
    name: str
    description: str | None = None
    display_name: str | None = None
    visibility_type: str | None = None
    semantic_type: str | None = None
    has_field_values: str | None = None
    coercion_strategy: str | None = None
    number_style: str | None = None
    decimals: int | None = None
    currency: str | None = None

    fk_target_table: str | None = None
    fk_target_field: str | None = None

    meta_fields: MutableMapping = dc.field(default_factory=dict)


@dc.dataclass
class Model:
    database: str
    schema: str
    group: Group

    name: str
    alias: str
    description: str | None = None
    display_name: str | None = None
    visibility_type: str | None = None
    points_of_interest: str | None = None
    caveats: str | None = None

    unique_id: str | None = None
    source: str | None = None
    tags: Sequence[str] | None = dc.field(default_factory=list)

    columns: Sequence[Column] = dc.field(default_factory=list)

    @property
    def ref(self) -> str | None:
        if self.group == Group.nodes:
            return f"ref('{self.name}')"
        elif self.group == Group.sources:
            return f"source('{self.source}', '{self.name}')"
        return None

    @property
    def alias_path(self) -> str:
        return ".".join(
            [
                self.database or DEFAULT_DATABASE,
                self.schema or DEFAULT_SCHEMA,
                self.alias,
            ]
        )

    def format_description(
        self,
        append_tags: bool = False,
        docs_url: str | None = None,
    ) -> str:
        """Formats description from available information.

        Args:
            append_tags (bool, optional): True to include dbt model tags. Defaults to False.
            docs_url (Optional[str], optional): Provide docs base URL to include links. Defaults to None.

        Returns:
            str: Formatted description.
        """

        sections = []

        if self.description:
            sections.append(self.description)

        if append_tags and self.tags:
            sections.append(f"Tags: {', '.join(self.tags)}")

        if docs_url:
            sections.append(
                f"dbt docs: {docs_url.rstrip('/')}/#!/model/{self.unique_id}"
            )

        return "\n\n".join(sections)
