import mlflow


class LogisticRegressionAdapter(mlflow.pyfunc.PythonModel):

    def __init__(self, trained_model):
        self._model = trained_model

    def predict(self, context, model_input):
        return self._model.predict_proba(model_input)[:, 1]
