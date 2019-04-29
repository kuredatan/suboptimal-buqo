# coding:utf-8

from scipy.optimize import minimize
from parameters import resize_image, iota, l2, sqrtM, pad_image, print_probas, M
import numpy as np
import proxmin.algorithms as pm
import proxmin.operators as op
from itertools import count

#-------------- MAP ESTIMATION

## TODO (for the "marguerite" picture)
def feasible_MAP(x):
	x[x < 1] = 1
	x[x > 180] = 180
	return x

def prox_step(x, step, grad_f=None, nb_step=None):
	next(nb_step)
	return x-step*grad_f(x)

## x_hat = argmin_x g1(x)+g2(x) using proximal gradient iteration
def estimation_MAP(y, sigma, P, g1, g2, N, Phi, epsilon, verbose=False):
	nb_step = count(0)
	grad_f = lambda x : -2*g1(x, y, sigma)*pad_image(P(x)-y)
	step_f = 1./(4.*(1.**2)*(float(epsilon)**2))
	prox_g = lambda x, step : minimize(lambda u : g2(u, Phi)+1/(2*step)*l2(x-u), x, method="nelder-mead", options={'disp': True, 'maxiter':10}).x
	prox_map = lambda x, step : prox_g(prox_step(x, step, grad_f, nb_step), step)
	x = feasible_MAP(pad_image(y))
	res = pm.pgm(x, prox_map, step_f, accelerated=False)
	print("Converged: " + str(res[0]) + " in " + str(nb_step) + " iterations.")
	print("Average error: " + str(np.mean(res[1])))
	return x

#-------------- PROJECTION ONTO CONVEX SETS S and C_alpha

## TODO (for the "marguerite" picture)
def prox_C(x, step, g2=None, Phi=None, x_hat=None, N=None, tau=None):
	## Feasible point
	x[x < 1] = 1
	x[x > 180] = 180
	thresh = g2(x_hat, Phi)+N*(tau+1)
	res = minimize(lambda u : iota(g2(u, Phi) <= thresh), x, method="nelder-mead", options={'disp': True, 'maxiter':10}).x
	return res

## TODO (for the "marguerite" picture): background removal
def prox_S(u, step):
	## Feasible point
	u[u < 0] = 0.
	u[u > 256] = 256.
	m = sqrtM//2
	u = resize_image(u)
	v = u[:m, :, :]
	v[v < 1] = 1
	v[v > 74] = 74
	u[:m, :, :] = v
	v = u[3*m:, :, :]
	v[v < 1] = 1
	v[v > 74] = 74
	u[3*m:, :, :] = v
	v = u[m:3*m, (3*m+1):, :]
	v[v < 1] = 1
	v[v > 74] = 74
	u[m:3*m, (3*m+1):, :] = v
	v = u[m:3*m, :m, :]
	v[v < 1] = 1
	v[v > 74] = 74
	u[m:3*m, :m, :] = v
	u = u.flatten()
	return u

def in_S(x):
	return l2(prox_S(x, 0.)-x) == 0

## Proximal gradient iteration
def projection_C(x, P, y, epsilon, g1, g2, sigma, Phi, x_hat, tau, N):
	print("\n>> Projection onto C_alpha")
	nb_step = count(0)
	grad_f = lambda u : x-u-2*g1(u, y, sigma)*pad_image(P(u)-y)
	step_f =  1./(1.+4.*(1.**2)*(float(epsilon)**2))
	prox_proj_C = lambda x, step : prox_C(prox_step(x, step, grad_f, nb_step), step, g2=g2, Phi=Phi, x_hat=x_hat, N=N, tau=tau)
	res = pm.pgm(x, prox_proj_C, step_f, accelerated=False)
	print("Converged: " + str(res[0]) + " in " + str(nb_step) + " iterations.")
	print("Average error: " + str(np.mean(res[1])))
	return x

def projection_S(x):
	print("\n>> Projection onto S")
	nb_step = count(0)
	grad_f = lambda u : x-u
	step_f =  1.
	prox_proj_S = lambda x, step : prox_S(prox_step(x, step, grad_f, nb_step), step)
	res = pm.pgm(x, prox_proj_S, step_f, accelerated=False)
	print("Converged: " + str(res[0]) + " in " + str(nb_step) + " iterations.")
	print("Average error: " + str(np.mean(res[1])))
	return x
