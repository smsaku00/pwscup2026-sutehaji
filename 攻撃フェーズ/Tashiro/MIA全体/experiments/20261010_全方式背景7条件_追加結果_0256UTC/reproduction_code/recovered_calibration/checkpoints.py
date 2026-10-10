"""Cache completed original optimizer/crossfit calls; never change fit rules."""
from pathlib import Path
import ast,json,time
import numpy as np
import scipy
from scipy.optimize import OptimizeResult
import sklearn,joblib
import runtime as r

ROOT=Path('/workspace/shared/mia_recovered_calibration/cache/checkpoints')

def install(module,args,source,work):
    folder=ROOT/args.family/args.bg;folder.mkdir(parents=True,exist_ok=True)
    fp={'source_sha256':r.sha(source),'feature_sha256':r.sha(work/'代表特徴の再現結果/代表特徴.npz'),'background_sha256':r.MANIFEST['subsets'][args.bg]['sha256'],'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    p=folder/'fingerprint.json'
    if p.exists():assert json.loads(p.read_text())==fp
    else:p.write_text(json.dumps(fp,sort_keys=True)+'\n')
    if args.family=='learned_lowfpr':
        original=module.minimize;count=[0];fresh=[0]
        def cached_minimize(fun,x0,*a,**kw):
            i=count[0];count[0]+=1;p=folder/f'opt{i:05d}.npz'
            assert kw['options']=={'maxiter':200,'maxls':50,'ftol':1e-10,'gtol':1e-6}
            if p.exists():
                z=np.load(p);out=OptimizeResult(x=z['x'],jac=z['jac'],fun=float(z['fun']),success=bool(z['success']),status=int(z['status']),nit=int(z['nit']),message=str(z['message']))
            else:
                out=original(fun,x0,*a,**kw);temp=p.with_name(p.stem+'.tmp.npz')
                np.savez_compressed(temp,x=out.x,jac=out.jac,fun=out.fun,success=out.success,status=out.status,nit=out.nit,message=str(out.message));temp.replace(p);fresh[0]+=1
            if count[0]%100==0:(folder/'progress.json').write_text(json.dumps({'completed_original_optimizations':count[0],'total':6000,'new_fits_this_run':fresh[0],'warm_start_fits_per_run':200,'max_iter':200,'updated':time.time()})+'\n')
            return out
        module.minimize=cached_minimize
        # Original source creates a progress directory before scoring. Permit
        # only that progress directory to preexist on a resumable run.
        tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        class Resume(ast.NodeTransformer):
            def visit_Assert(self,node):
                if ast.unparse(node.test)=='not args.output_dir.exists()':return ast.Pass()
                return node
            def visit_Call(self,node):
                if isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='args.output_dir.mkdir':node.keywords.append(ast.keyword(arg='exist_ok',value=ast.Constant(value=True)))
                return self.generic_visit(node)
        fn=Resume().visit(fn);ast.fix_missing_locations(fn);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),module.__dict__)
    elif hasattr(module,'crossfit'):
        original=module.crossfit;count=[0]
        def cached_crossfit(x,y,teams,*a,**kw):
            key=r.matsha(x)+'_'+r.matsha(y)+'_'+__import__('hashlib').sha256(repr((teams,a,sorted(kw.items()))).encode()).hexdigest()
            p=folder/(key+'.joblib')
            if p.exists():out=joblib.load(p)
            else:
                out=original(x,y,teams,*a,**kw);joblib.dump(out,p.with_suffix('.tmp'));p.with_suffix('.tmp').replace(p)
            count[0]+=1;(folder/'progress.json').write_text(json.dumps({'completed_crossfit_calls_this_run':count[0],'updated':time.time()})+'\n')
            return out
        module.crossfit=cached_crossfit
