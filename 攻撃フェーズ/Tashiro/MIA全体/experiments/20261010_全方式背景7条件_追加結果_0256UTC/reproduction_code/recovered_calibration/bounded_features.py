"""Memory-bounded evaluation of row-independent original BG reference terms."""
import ast
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

def install(module,source):
    def bounded_reference(a,q,ar7,x7,h,block=256):
        freq={}
        for col in module.u.CAT7:
            levels=sorted(set(a[col])|set(q[col]))
            freq[col]=(a[col].value_counts().reindex(levels,fill_value=0)+1)/(len(a)+len(levels))
        near=[];density=[]
        for start in range(0,len(q),block):
            frame=q.iloc[start:start+block];distance=cdist(x7[start:start+block],ar7,'sqeuclidean');d=distance.copy()
            for col in module.u.CAT7:d+=4*(frame[col].to_numpy()[:,None]!=a[col].to_numpy()[None,:])
            near.append(module.nearest(d,3).mean(axis=1));del d
            result=-.5*distance/h**2
            for col in module.u.CAT7:
                eq=frame[col].to_numpy()[:,None]==a[col].to_numpy()[None,:]
                p=frame[col].map(freq[col]).to_numpy()[:,None]
                result+=np.log(.98*eq+.02*p)
            density.append(logsumexp(result,axis=1)-np.log(len(a)))
        return np.concatenate(near),np.concatenate(density)
    module.bounded_reference=bounded_reference
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calculate')
    def assigns(node,name):return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    start=next(i for i,n in enumerate(fn.body) if assigns(n,'da7'))
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Delete) and any(isinstance(t,ast.Name) and t.id=='da7' for t in n.targets))
    h=next(n for n in fn.body[start:end+1] if assigns(n,'h'))
    replacement=ast.parse('na7,log_a=bounded_reference(a,q,ar7,x7,h)').body[0]
    fn.body=fn.body[:start]+[h,replacement]+fn.body[end+1:]
    ast.fix_missing_locations(fn);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),module.__dict__)
    return bounded_reference
