def extract_x_y(data):
    features = data.drop(["situacao"], axis = 1)
    target = data.situacao
    
    return features, target

def plot_pie_imba(data):
    data_by_situacao = data[["situacao"]].groupby(["situacao"]).size()
    data_by_situacao.plot(kind = "pie")

def evaluate_model(model, train_features, validation_features):
    train_prediction = model.predict(train_features)
    validation_prediction = model.predict(validation_features)
    train_prediction_score = model.predict_proba(train_features)[:, 1]
    val_pred_score = ab_validation_y_score = model.predict_proba(validation_features)[:, 1]
    
    return {"train_pred": train_prediction, "train_pred_score": train_prediction_score, "val_pred": validation_prediction, "val_pred_score": val_pred_score} 