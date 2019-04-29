#coding:utf-8

from parameters import get_image
from main import main
import sys
import dill as pickle

def robustness(x):
	true_sigma = 5
	sigma_values = range(2, 20, 1)
	rhos, vs = [], []
	for sigma in sigma_values:
		print(">> Sigma = " + str(sigma))
		res = main(x, sigma=sigma, true_sigma=true_sigma, print_stuff=False)
		rhos.append(res[0])
		vs.append(res[1])
		with open("intermediary_robustness.pi", "wb") as f:
			pickle.dump({'nombre': vs, 'rhos': rhos, 'sigma_values': sigma_values, 'current_sigma': sigma}, f)
	plt.plot([sigma-true_sigma for sigma in sigma_values], rhos, col=["g+" if (v==1) else "r+" for v in vs], title="Evolution de rho_alpha\nen fonction de sigma-true_sigma\n(true_sigma=" + str(true_sigma)+")")
	plt.savefig("robustness.jpg", bbox_inches="tight")
	plt.show()

def calibration(x):
	niter = 100
	vs = []
	sigma = 5
	for i in range(niter):
		print(">> Iteration #" + str(i+1))
		res = main(x, sigma=sigma, print_stuff=False, niter=1)
		vs.append(res[1])
		with open("intermediary_calibration.pi", "wb") as f:
			pickle.dump({'nombre': vs, 'niter': i}, f)
	print("Nombre empirique :", round(100*sum([v[0] for v in vs])/niter,1))
	print("Valeur théorique :", round(100*alpha, 1))
	with open("calibration.pi", "wb") as f:
		pickle.dump({'empirique': 100*sum(vs)/niter, 'theorique': 100*alpha, 'niter': niter}, f)

if __name__ == '__main__':
	di = {'robustness': robustness, 'calibration': calibration}
	x = get_image()
	if (sys.argv[1] != ""):
		di[sys.argv[1]](x)
	else:
		print("Ecrire le nom du test.")
