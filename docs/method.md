# Method

## The interior flux

Let the star be the unit disk, $r$ the distance from its centre and $\mu=\sqrt{1-r^2}$. Every
supported law is a sum of powers,
$I(\mu)=\sum_i c_i\,\mu^{\alpha_i}$. For an occulting disk of radius $p$ at separation $z$ that lies
entirely inside the star ($z+p\le1$),

$$
F = 1-\frac{p^2}{2}\,\frac{\sum_i c_i\,D_{\alpha_i/2}(z,p)}{\sum_i c_i/(\alpha_i+2)},
\qquad
D_\gamma(z,p)=\frac{1}{\pi p^2}\int_{\rm disk}(1-r^2)^\gamma\,\mathrm dA .
$$

For integer $\gamma$, $D_\gamma$ is a polynomial: $D_0=1$ and $D_1=1-z^2-p^2/2$. The quadratic law
needs only one non-elementary mean, $D_{1/2}=\langle\mu\rangle$. In the closed form of Mandel & Agol
(2002) this mean requires complete elliptic integrals of all three kinds.

## The series

Write a point of the occulting disk as $(z+\rho\cos\phi,\rho\sin\phi)$. Then
$1-r^2=w(1-\varepsilon)$ with $w=1-z^2$ and $\varepsilon=(2z\rho\cos\phi+\rho^2)/w$, and
$|\varepsilon|\le e=(2zp+p^2)/w<1$ inside the star. With $\beta_m=(-1)^m\binom{\gamma}{m}$ the
binomial series converges uniformly, and its disk averages are elementary:

$$
D_\gamma = w^\gamma\sum_{m\ge0}\beta_m\langle\varepsilon^m\rangle,\qquad
\langle\varepsilon^m\rangle = v^m\sum_{l\ \rm even}G_{m,l}\,z^l p^{2m-l}\ \ge 0,
$$

with $v=1/w$ and $G_{m,l}=\binom{m}{l}\binom{l}{l/2}\frac{2}{2m-l+2}$.

Truncated after $m=M=11$, this is $w^\gamma\sum_{j=0}^{M}a_j(p)\,v^j$. The coefficients $a_j(p)$
are computed once per light curve. For the quadratic law each in-transit point costs

$$
F \simeq A_0 + A_1 z^2 - \sqrt{1-z^2}\sum_{j=0}^{11}\tilde a_j\,v^j ,
$$

which is one square root, one division and about a dozen multiply–adds. The expression has no
branches and no special cases at $z=0$ or $z=p$.

The series is the two-dimensional form of Pizzetti's (1909) mean-value expansion,
$\langle f\rangle = \sum_n \frac{(p^2/4)^n}{n!(n+1)!}\Delta^n f(z)$. The classical small-planet
approximation is its first term. The qpower2 algorithm (Maxted & Gill 2019) is its first two terms.

## The error bound

For $m>\gamma$ all $\beta_m$ have the same sign and $|\beta_m|$ does not increase, while all
$\langle\varepsilon^m\rangle\ge0$. The remainder therefore has a known sign, and

$$
|S_M-D_\gamma|\ \le\ B_M = \frac{w^\gamma\,|\beta_{M+1}|\,\langle\varepsilon^{M+1}\rangle}{1-e}
\qquad(M\ \text{odd},\ M+1>\gamma).
$$

$B_M$ increases with $z$. `pizzetti` therefore finds, once per light curve, the largest $z_c$ at
which the flux bound $\frac{p^2}{2}\sum_i|c_i|B_M(\gamma_i)/|\sum_i c_i/(\alpha_i+2)|$ equals
`series_tol`. Every point with $z\le z_c$ is then guaranteed to be within `series_tol` of the
exact flux. In practice the bound exceeds the true error by a factor of only 1.0–1.6 over most of
the interior.

The bound diverges at second contact ($z\to1-p$), where the disk mean has a
$\delta^2\ln\delta$ singularity. The series therefore covers 96–98 % of an Earth-size transit and
55–75 % of a Jupiter-size transit at the default tolerance.

## Near and across the limb

Points with $z_c<z<1+p$ use a reference algorithm:

- **Uniform, linear and quadratic laws:** the exact Mandel & Agol (2002) solution. It is evaluated
  with Bulirsch's general complete elliptic integral at a fixed five steps in a vectorised loop,
  and the elliptic arguments are computed in closed form from $(z,p)$ (Agol, Luger &
  Foreman-Mackey 2020). The rare unconverged lanes fall back to an early-exit scalar iteration.
  The arccos and area terms use factored forms whose cancelling sums are exact, so the solution
  stays accurate to ~10⁻¹⁴ even within 10⁻¹⁵ of the contact points.
- **Square-root, nonlinear and power-2 laws:** Green's theorem with the radial primitive
  $F(r)=\int_0^r(1-s^2)^\gamma s\,\mathrm ds$. The blocked flux is a closed-form term on the stellar
  limb plus a one-dimensional integral along the occultor's rim inside the star. The integral is
  evaluated by composite Gauss–Legendre quadrature, with segments that grow geometrically away from
  the nearest singularity of $(1-r^2)^{\gamma+1}$ and a square-root substitution at the limb
  crossing. The accuracy is ≤ 3 × 10⁻¹¹.

A complete light curve therefore combines the series, for the bulk of the transit, with one of
these reference algorithms for ingress and egress.
