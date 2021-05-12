import numpy as np

def lstm_pred(model, true, p, pred_steps):
    pred = []
    X_tst = np.expand_dims(true[:p], axis = 0).copy()
    for j in range(pred_steps): 
        pre = model.predict(X_tst)
        pred.append(pre)
        X_tst[0, :, :] = np.concatenate((X_tst[0, 1:, :], pre), axis = 0)
    pred = np.array(pred).squeeze()
    pred = np.concatenate((true[:p], pred))
    return pred