#coding:utf-8

from parameters import selection_phi, noise, P, print_probas, print_image, get_image, g1, g2, l2, iota, N, alpha, compute_epsilon, compute_tau, characteristic
from tools import estimation_MAP, projection_C, projection_S, in_S
import numpy as np

#-------------- BUQO ALGORITHM

def main(x, sigma=5, niter=1, true_sigma=None, Phi=None, eps = 1e-5, print_stuff=True):
	rhos, vs = [], []
	if (not true_sigma):
		true_sigma = sigma
	epsilon = compute_epsilon(sigma)
	tau = compute_tau(alpha)
	for _ in range(niter):
		Phi = selection_phi(x) if (not Phi) else Phi
		print("Phi = " + str(Phi))
		## Preprocessing: turn x into the observation y
		print("\n>>>>>>>>> Observation")
		y = P(x) + noise(sigma=true_sigma)
		if (print_stuff):
			print("[True value] p(x|y) = " + str(round(np.exp(-g1(x, y, sigma)-g2(x, Phi)), 5)))
			print_image(y, "Observation of true image")
		## I. MAP estimation
		print("\n>>>>>>>>> MAP estimation")
		x_hat = estimation_MAP(y, sigma, P, g1, g2, N, Phi, epsilon)
		if (print_stuff):
			print_probas(x_hat, Phi, y, sigma, var="x_hat")
			print_image(x_hat, "MAP estimation")
		## II. Define S (see tools.py)
		## III. Compute C_alpha (see tools.py)
		## IV. Compute dist(S, C_alpha)
		print("\n>>>>>>>>> Distance C_alpha - S")
		## Alternating projection algorithm (Von Neumann's)
		f_C = lambda u : g1(u, y, sigma)+iota(g2(x, Phi) <= g2(x_hat, Phi)+N*(tau+1))
		f_S = lambda u : iota(in_S(u))
		## Random initialization for POCS
		x0 = np.random.normal(50, 50, len(x_hat))
		if (print_stuff):
			print_image(x0, "Image initiale")
		y_old = projection_C(x0, P, y, epsilon, g1, g2, sigma, Phi, x_hat, tau, N)
		x_old = projection_S(y_old.copy())
		i = 0
		niter = 1
		while (i < niter):
			y_s = projection_C(x_old.copy(), P, y, epsilon, g1, g2, sigma, Phi, x_hat, tau, N)
			x_s = projection_S(y_s.copy())
			if (l2(x_s-x_old) < eps or l2(y_s-y_old) < eps):
				print("\n(*) Converged: True in " + str(i+1) + " iterations.")
				break
			i += 1
			y_old = y_s.copy()
			x_old = x_s.copy()
		if (not niter):
			y_s = y_old.copy()
			x_s = x_old.copy()
		if (i == niter):
			print("Warning: Maximum number of iterations (niter=" + str(niter) + ") has been exceeded.")
		d = l2(x_s-y_s)
		if (print_stuff):
			print("\n    | f_C     | f_S")
			print("----|---------|-----")
			print("y_C | " + str(f_C(y_s)) + "     | " + str(f_S(y_s)))
			print("\^x | " + str(f_C(x_hat)) + "     | " + str(f_S(x_hat)))
			print("x_S | " + str(f_C(x_s)) + "     | " + str(f_S(x_s)) + "\n")
			print("d(C_alpha, S) = " + str(round(d, 3)))
			print_probas(x_s, Phi, y, sigma, var="x_S")
			print_image(x_s, "Argmin_{x \in S} d(x, C_alpha)")
		print("\n>>>>>>>>> Statistical test")
		rho_alpha = d/l2(x_hat-x_s)
		## V. "Statistical" test
		if (d > 0):
			print("REJECT H0 with alpha = " + str(alpha))
			v = 1
		else:
			print("CANNOT REJECT H0 with alpha = " + str(alpha))
			v = 0
		print("rho_alpha = " + str(round(rho_alpha, 2)))
		print("----------------------------------------")
		rhos.append(rho_alpha)
		vs.append(v)
	return rhos, vs

#-------------- TESTS

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	niter = 1
	x = get_image()
	rhos, vs = main(x, niter=niter, print_stuff=True)
	if (niter > 1):
		plt.plot(range(niter), rhos, col=["g+" if (v==1) else "r+" for v in vs])
	plt.show()
