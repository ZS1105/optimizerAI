# optimizerAI
# 机器学习及优化算法
工具箱为每个优化算法提供统一的参数接口，因此使用的时候只需要关注自己的目标函数即可。使用仅需四个步骤：
  1、定义目标函数；
  2、设置目标函数待优化参数信息；
  3、设置优化器本身需要的参数信息；
  4、调用优化算法。
  
1.定义目标函数：
  from sklearn.svm import SVC
  from sklearn import metrics

  def svc_objf(C_gamma, Xtrain=None, Ytrain=None, Xtest=None, Ytest=None):
   
    '''
    构造SVM分类模型目标函数（适应度函数）
    '''
    mdl = SVC(C=C_gamma[0], gamma=C_gamma[1])
    mdl = mdl.fit(Xtrain, Ytrain)
    Ypre = mdl.predict(Xtest)
    error = 1 - metrics.accuracy_score(Ytest, Ypre)
    
    return error

2. 设置目标函数中待优化参数信息，假设我们x为100维，每个维度在[-10， 10]范围内取值，则目标函数参数信息设置为：
  parms_func = {'x_lb': 0.01, 'x_ub': 100, 'dim': 2,
		'kwargs': {'Xtrain': Xtrain, 'Ytrain': Ytrain,
			  'Xtest': Xtest, 'Ytest': Ytest}}
  
3. 设置优化器需要的参数信息：
  parms_gwo = {'opter_name': 'GWO', 'PopSize': 20, 'Niter': 100}

4. 整合参数信息并调用优化器（GWO）就可以了完成优化：
  func_opter_parms = FuncOpterInfo(parms_func, parms_opter)
  func_opter_parms = GWO(svc_objf, func_opter_parms)

5. 查看最优参数结果：
  gwo_parms.best_x
