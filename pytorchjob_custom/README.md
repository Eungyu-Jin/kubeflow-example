# pytorchjob training-operator used in kfp-v2

kubeflow git에 있는 분산 학습을 위한 pytorchjob은 v1에 최적화 되어 있어, kfp-v2에는 작동이 안 됨  
kfp-v2에 pipeline task로 실해될 수 있도록 pyotchjob task를 custom으로 구성
