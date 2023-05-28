from sim_utils import *
from scipy.optimize import minimize

vec_array=get_lcdev_vecs()
sim_set=create_simset(vec_array)
print('sim_set example:   ',sim_set[0])

def opfunc(x):#输入值为x即相似度百分比切割线的值，得出的结果：acc=(超过x的相似度为1的结果num+小于x的相似度为0结果num)/总num，并打印假真和真假所占百分比
	tt,tf,ft,ff=0,0,0,0
	total=len(sim_set)
	for i in sim_set:
		if((i[0] >= x[0]) and (i[1]=='1')):
			tt+=1
		elif((i[0] < x[0]) and (i[1]=='0')):
			ff+=1
		elif((i[0] >= x[0]) and (i[1]=='0')):#以为是t结果是f
			tf+=1
		elif((i[0] < x[0]) and (i[1]=='1')):
			ft+=1
	return 1-(tt+ff)/total#最小化所以精度是1-res值

res=minimize(opfunc, numpy.asarray((0.5)), method='Nelder-Mead')#直接单纯形法，不能带cons，建议函数内部写好
print('res.success',res.success)
print('res.x',res.x)
print('res.fun',res.fun)

print(opfunc([0.85292969]))

#        任       务,     最佳x值   ，最佳结果
# oppo_train     集,0.87500000,0.24555998875416485
# oppo_dev      集,0.87822266,0.2558
#trainx-devfunc  ,0.87500000,0.2563
#lcqmc_train      ,0.85292969,0.12691086670631502
#lcqmc_ dev      ,0.89375000,0.19972733469665982
#trainx-devfunc  ,0.85292969,0.2185866848443535
