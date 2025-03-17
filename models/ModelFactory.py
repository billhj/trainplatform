import importlib

from models.LinearRegressionModel import LinearRegressionModel


def load_model(model_name: str):
    try:
        print(model_name)
        module = importlib.import_module(model_name)
        model_class = getattr(module, model_name)
        return model_class()
    except (ImportError, AttributeError) as e:
        # 如果导入失败，抛出 HTTPException。
        print(f"ImportError: {e}")
        #raise HTTPException(status_code=404, detail="Model not found") from e



#model = load_model('LinearRegressionModel')
#print(model.default_params)

modelnames = ['LinearRegressionModel', 'LogisticRegressionModel']


class ModelFactory:
    def __init__(self):
        self.models = {}

    def load_models(self):
        for model_name in modelnames:
            model = load_model(model_name)
            if model is not None:
                self.models[model_name] = model
                print("Loading model:", model_name)
                self.models[model_name] = model

    def load_models_by_names(self, model_names):
        for model_name in model_names:
            model = load_model(model_name)
            if model is not None:
                self.models[model_name] = model
                print("Loading model:", model_name)
                self.models[model_name] = model

    def get_model(self, model_name):
        return self.models[model_name]

    def get_models(self):
        return self.models

#mk = ModelFactory()
#mk.load_models()
