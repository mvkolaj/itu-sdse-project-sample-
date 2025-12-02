# %% [markdown]
# # DEPLOY
# 
# A model version can be assigned to one or more stages. MLflow provides predefined stages for common use cases: None, Staging, Production, and Archived. With the necessary permissions, you can transition a model version between stages or request a transition to a different stage.

# %%
model_version = 1

# %% [markdown]
# # Transition to staging

# %%
from mlflow.tracking import MlflowClient

client = MlflowClient()


def wait_for_deployment(model_name, model_version, stage='Staging'):
    status = False
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name,version=model_version)
            )
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    return status

model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
model_status = True
if model_version_details['current_stage'] != 'Staging':
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,stage="Staging", 
        archive_existing_versions=True
    )
    model_status = wait_for_deployment(model_name, model_version, 'Staging')
else:
    print('Model already in staging')


