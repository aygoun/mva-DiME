# Upload the models to HuggingFace
/root/.local/bin/hf upload armandblin/DiME_audio ./models/ \
  --repo-type model \
  --include "*" \
  --commit-message "Upload audio models"
