import mlflow
import mlflow.tensorflow
from mlflow.tracking.client import MlflowClient


mlflow.set_experiment("/MLFlow")
mlflow.tensorflow.autolog()


with mlflow.start_run(run_name='training_001') as run:
    history = model.fit(X_train, y_train_cat, 
                        batch_size = 16, 
                        verbose=1, 
                        epochs=50, 
                        validation_data=(X_test, y_test_cat), 
                        #class_weight=class_weights,
                        # callbacks=[callback],
                        shuffle=False)

    _, acc = model.evaluate(X_test, y_test_cat)
    print("Accuracy is = ", (acc * 100.0), "%")
    
    
    
# model_version = 2
# client = MlflowClient()
# client.update_model_version(
# name=model_name,
# version=model_version,
# description="This is the best model version 2. It's a U-Net model with CNNs layers"
# )



# Run script to change current model to production.
stage = "Production"
client.transition_model_version_stage(
name=model_name,
version=model_version,
stage=stage,
)
