import wandb
api = wandb.Api()

# run = api.run("<entity>/<project>/<run_id>")
# run.config["key"] = updated_value
# run.update()
api.runs("project_path_here", filters={"createdAt": {"$in": ["date"]}})
runs = api.runs("q191106702/CVR prediction test", filters={"tags": {"$in": ["src.models.common.Basic_Loss"]}})

for run in runs:
    run.tags = ['src.models.common.Entire_Space_Basic_Loss' if tag == 'src.models.common.Basic_Loss' else tag for tag in run.tags]
    print(run.tags)
    run.update()
