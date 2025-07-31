# Build the index
reposage index

# Query
reposage query "What is DeepSeek?" --threshold 0.3

# Or call the REST API
curl -X POST https://<your-space>.hf.space/query \
     -d '{"question":"What is DeepSeek?"}'
