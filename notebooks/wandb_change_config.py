import wandb
api = wandb.Api()

# run = api.run("<entity>/<project>/<run_id>")
# run.config["key"] = updated_value
# run.update()

run = api.run("q191106702/CVR prediction/rvx3fza4")
run.group = "CGAM-IPW-hard_mining-contrastive_loss_proportion"
run.update()