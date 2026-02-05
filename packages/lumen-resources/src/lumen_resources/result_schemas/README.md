```bash
datamodel-codegen \
  --input src/lumen_resources/schemas/result_schemas \
  --output src/lumen_resources/result_schemas/ \
  --use-schema-description \
  --use-field-description \
  --target-python-version 3.10 \
  --use-standard-collections \
  --use-union-operator \
  --output-model-type pydantic_v2.BaseModel \
  --field-constraints \
  --input-file-type jsonschema \
  --allow-population-by-field-name
```

```bash
datamodel-codegen \
  --input src/lumen_resources/schemas/config-schema.yaml \
  --output src/lumen_resources/lumen_config.py \
  --use-schema-description \
  --use-field-description \
  --target-python-version 3.10 \
  --use-standard-collections \
  --use-union-operator \
  --output-model-type pydantic_v2.BaseModel \
  --field-constraints \
  --input-file-type jsonschema \
  --allow-population-by-field-name
```
