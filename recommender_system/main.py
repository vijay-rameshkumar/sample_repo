import subprocess, os

def train():
    scripts1 = 'python recommender_system\scripts\data_analysis_preparation.py'
    scripts2 = 'python recommender_system\scripts\data_feaure_engineering.py'
    scripts3 = 'python recommender_system\scripts\data_model_train.py'

    scripts = [scripts1, scripts2, scripts3]

    for script in scripts:
        p = subprocess.Popen(script, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        out = out.decode('utf-8')
        result = out.split('\n')
        print(out)

        for lin in result:
            if not lin.startswith('#'):
                print(lin)

if __name__ == '__main__':
    train()