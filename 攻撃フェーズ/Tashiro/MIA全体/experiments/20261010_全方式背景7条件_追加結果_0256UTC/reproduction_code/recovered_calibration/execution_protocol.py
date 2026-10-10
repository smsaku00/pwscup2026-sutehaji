"""Deterministic accumulation of an unchanged fitted forest's tree predictions."""
PROTOCOL='fixed_tree_order_serial_predict_v1'

def install():
    from sklearn.ensemble._forest import ForestClassifier
    if getattr(ForestClassifier.predict_proba,'_bg7_serial',False):return
    original=ForestClassifier.predict_proba
    def predict_proba(self,X):
        previous=self.n_jobs;self.n_jobs=1
        try:return original(self,X)
        finally:self.n_jobs=previous
    predict_proba._bg7_serial=True
    predict_proba._original=original
    ForestClassifier.predict_proba=predict_proba
