# bullet-manipulation

This repo contains PyBullet-based manipulation environments consisting of a Sawyer robot manipulating drawers and picking and placing objects.


<img src="https://lh6.googleusercontent.com/XRlXh_ShnwT4fG0d7AqAXZVmUO0DpmjbqhBiAxGEpJATq1_MrrUyVVytpy5KKNGSXAhNfyDNIGTgxRQwuLdEbJk=w1280" width="200" />
<img src="https://lh5.googleusercontent.com/wfSuDGSeluN8gC4GYf5KQUhtt3nJFJbL0QNF7vb-wA7sqflfNR29CMYPP89_TOlVzR55K3heE_6PUl-HGpkQuY8=w1280" width="200" />
<img src="https://lh6.googleusercontent.com/5g49wEN3MJQRNq4I1xm35ILVFYszgBUrRX79tydPa7_pJRbQRNdOEWMRTBaKCNLk4f1u6KCwpjHNCUhrwJwvVNQ=w1280" width="200" />

<img src="https://lh5.googleusercontent.com/BWPLSAOgz9ZhNyOX07VqFSWkK7XS-AUeYWKTmB-Sj5L8nppDfsqkT4Ek6tUO8jYzONfdQo8yiO_x5NWFyh6EJwU=w1280" width="200" />
<img src="https://lh6.googleusercontent.com/R78OUpKzUbqL76pWIc3hUKsazMhww95j8-GDE2b3iRVSUjJupcDqF_6Z_I7cLMya6BKtIXgRSvom1HK1PQR59BM=w1280" width="200" />
<img src="https://lh4.googleusercontent.com/so1ULj8BFb6SlAUqQLXi1JWC_OTj6_BNiCQz1W83StUDf-xHkr-vGO9qp8VRfWmwstt36-l9HgGl07r0Z_sKb9k=w1280" width="200" />

These environments are featured in:
[What Can I Do Here? Learning New Skills by Imagining Visual Affordances](https://arxiv.org/abs/2106.00671)
Alexander Khazatsky*, Ashvin Nair*, Daniel Jing, Sergey Levine. International Conference on Robotics and Automation (ICRA), 2021.
[Offline Meta-Reinforcement Learning with Online Self-Supervision](https://arxiv.org/abs/2107.03974)
Vitchyr H. Pong, Ashvin Nair, Laura Smith, Catherine Huang, Sergey Levine. arXiv preprint, 2021.

This repository extends https://github.com/avisingh599/roboverse which was developed by Avi Singh, Albert Yu, Jonathan Yang, Michael Janner, Huihan Liu, and Gaoyue Zhou.

## Setup
`pip install -r requirements.txt`

## Quick start
The best way to get started is to run:

```
python scripts/collect_data.py --gui True
```

If you have a spacemouse, you can run the environments interactively with: 
```
python scripts/spacemouse_control.py
```

