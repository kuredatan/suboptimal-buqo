#coding:utf-8

from scipy.optimize import minimize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

## Fixed parameters
sqrtN = 80
N = sqrtN*sqrtN*3
sqrtM = 40
M = sqrtM*sqrtM*3
ratioMN = M/N
alpha = 0.05

## Shortcuts
l1 = lambda x : np.linalg.norm(x, 1)
l2 = lambda x : np.linalg.norm(x, 2)
iota = lambda c : np.inf if (not c) else 0
characteristic = lambda c : 1 if (c) else 0
eucl_ball = lambda x, center, radius : characteristic(l2(x-center) < radius)
compute_epsilon = lambda sigma : sigma*np.sqrt(M*(4+2*np.sqrt(M)))
compute_tau = lambda alpha : np.sqrt(16*np.log(3/float(alpha))/float(N))

#-------------- IMAGE RELATED

def resize_image(im):
	n = np.shape(im)[0]//3
	n = int(np.sqrt(n))
	return np.reshape(np.asarray(im, dtype=np.uint8), [n, n, 3])

def print_image(im, title=""):
	plt.imshow(resize_image(im))
	plt.axis('off')
	plt.title(title)
	plt.show()

def get_image(path="marguerite.jpg"):
	x = np.array(Image.open("images/"+path).resize([sqrtN,sqrtN]), dtype=float).flatten()
	return x

def pad_image(y):
	x = np.zeros((sqrtN, sqrtN, 3))
	m = sqrtM//2
	y = np.reshape(y, [sqrtM, sqrtM, 3])
	x[m:3*m, m:3*m, :] = y
	x[x < 1] = 1
	x[x > 180] = 180
	return x.flatten()

#-------------- DATA MODEL RELATED

## Noise function (Gaussian white noise)
def noise(sigma=1):
	return np.random.normal(0, sigma, M)

## Measurement operator
# TODO (for the "marguerite" picture)
def P(x):
	m = sqrtM//2
	x = resize_image(x)
	y = x[m:3*m, m:3*m, :]
	return y.flatten()

#-------------- BAYESIAN MODEL RELATED

## Likelihood function (associated with Gaussian white noise)
g1 = lambda x, y, sigma : np.exp(-l2(P(x)-y)**2/float(2*sigma**2))

## Prior on original image
# TODO (for the "marguerite" picture)
## C = { x | for all pixel p in x, color(p) >= 1 && color(p) <= 180} 
## thus being nonempy closed convex as an intersection of halfspaces 
lambda_ = lambda _ : float(M/(N*281577.))
f = lambda x : lambda_(0)*l1(x)
g2 = lambda x, Phi : f(Phi*x)+iota((x >= 1).all() and (x <= 180).all())

def print_probas(x, Phi, y, sigma, var="x", last=False):
	if (not last):
		print("* -log(p("+var+")) = " + str(round(g2(x, Phi), 5)))
		print("* -log(p(y|"+var+")) = " + str(round(g1(x, y, sigma), 5)))
		print("* Prior: p("+var+") ~ " + str(round(np.exp(-g2(x, Phi)), 5)))
		print("* Likelihood: p(y|"+var+") ~ " + str(round(np.exp(-g1(x, y, sigma)), 5)))
	if (last):
		print("  " + "_"*42)
	print("* Posterior distribution: p("+var+"|y) ~ " + str(round(np.exp(-g1(x, y, sigma)-g2(x, Phi)), 5)))

## Select the Phi that maximizes the prior probability p(x) when x = true picture
def selection_phi(x, eps=1e-5):
	f = lambda phi : g2(x, phi) + iota(abs(phi) > eps)
	phi0 = 10.0
	res = minimize(f, phi0, method="nelder-mead", options={'disp': False, 'xtol': 1e-5})
	return float(res.x[0])

#-------------- TESTS

if __name__ == '__main__':
	sigma = 5
	x = get_image()
	y = P(x)+noise(sigma)
	Phi = selection_phi(x)
	print_image(x, "True image")
	print_image(y, "Observation with sigma = " + str(sigma))
	print("(regularization value) Lambda = " + str(l1(P(x))))
	bg = x-pad_image(P(x))
	print_image(bg)
	print("Background color values = MAX=" + str(np.max(bg[bg > 0])) + " MIN=" + str(np.min(bg[bg > 0])))
	print_probas(x, Phi, y, sigma, var="x")
	#for a in [i*0.01 for i in range(1, 100)]:
	#	print(compute_tau(a))
