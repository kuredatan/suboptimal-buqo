La méthode BUQO a été conçue et implémentée originellement par Repetti et al. (voir référence de l'article ci-dessus).

```
@article{repetti2019scalable,
  title={Scalable Bayesian uncertainty quantification in imaging inverse problems via convex optimization},
  author={Repetti, Audrey and Pereyra, Marcelo and Wiaux, Yves},
  journal={SIAM Journal on Imaging Sciences},
  volume={12},
  number={1},
  pages={87--118},
  year={2019},
  publisher={SIAM}
}
```

Ceci est une version Python et **sous-optimale** de la méthode. A utiliser à vos risques et vos périls.

J'ai utilisé le package **proxmin** :

```
@article{proxmin,
    author="{Moolekamp}, Fred and {Melchior}, Peter",
    title="Block-simultaneous direction method of multipliers: a proximal primal-dual splitting algorithm for nonconvex problems with multiple constraints",
    journal="Optimization and Engineering",
    year="2018",
    month="Dec",
    volume=19,
    issue=4,
    pages={871-885},
    doi="10.1007/s11081-018-9380-y",
    url="https://doi.org/10.1007/s11081-018-9380-y",
    archivePrefix="arXiv",
    eprint={1708.09066},
    primaryClass="math.OC",
}
```

Télécharger les paquets nécessaires avec **pip** :

```bash
pip install -r requirements.txt
```

