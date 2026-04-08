from huggingface_hub import HfApi

TOKEN = input("Paste your HF token: ")
REPO  = "CHANDRAJITk/ai-query-routing"
api   = HfApi()

files = [
    "inference.py",
    "models.py",
    "openenv.yaml",
    "tasks/grader.py",
    "server/app.py",
    "server/ai_query_routing_environment.py",
]

for filepath in files:
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=filepath,
        repo_id=REPO,
        repo_type="space",
        token=TOKEN,
    )
    print(f"Uploaded {filepath}")

print("Done! Check HF Space.")
