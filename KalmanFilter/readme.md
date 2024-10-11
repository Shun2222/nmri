# Readme
## 軸について
方位$\phi$をNを基準とした時計回りであり，<br>
Nをx軸，Eをy軸とすると，$\theta = \pi/2 - \phi$となり，<br>
$\sin(\phi) = \cos(\theta), \cos(\phi) = \sin(\theta)$であるため，<br>
方位の値を$\sin$でN方向，$\cos$でE方向になる <br>
NE方向を固有値方向$\psi$に変換するには$F_1 = N\cdot \cos(\psi _1) + E\cdot \sin(\psi _1)$
## 船単位のデータで偏流計算
$A = \sum\sin(Hdg_i)^2$ <br>
$B = \sum-\sin(Hdg_i)\cos(Hdg_i)$ <br>
$C = \sum\cos(Hdg_i)^2$ <br>
$D = \sum(\sin(Hdg_i)\cos(Hdg_i)VogE_i - \sin(Hdg_i)^2VogN_i)$ <br>
$E = \sum(\sin(Hdg_i)\cos(Hdg_i)VogN_i - \cos(Hdg_i)^2VogE_i)$ <br>
$F = \sum (VogN_i\sin(Hdg_i)-VogE_i\cos(Hdg_i))^2$ <br>

### 固有値方向の対地ベクトルの寄与（VogをNE方向でなく，固有値方向に変換）
$θ1Rad = (Math.Abs(C - A) < 1e-08) ? Math.PI / 2 : Math.Atan2(-2 * B, C - A) / 2$ <br>
$θ2Rad = θ1Rad + Math.PI / 2$ <br>

$\lambda _1 = (Math.Abs(Cosθ1) > Math.Abs(Sinθ1)) ? A + B * Sinθ1 / Cosθ1 : B * Cosθ1 / Sinθ1 + C$ <br>
$\lambda _2 = (Math.Abs(Cosθ2) > Math.Abs(Sinθ2)) ? A + B * Sinθ2 / Cosθ2 : B * Cosθ2 / Sinθ2 + C$ <br>
$F1 = -(D\cos(\theta _1) + E\sin(\theta _1))/\lambda _1$ <br>
$F2 = -(D\cos(\theta _2) + E\sin(\theta _2))/\lambda _2$ <br>

## 全船の偏流を重みづけで計算
$w = \lambda \times \exp(-\log 2 \times \sqrt{a\times \Delta lat^2 + b\times\Delta lon^2 + c\times\Delta time^2})$ <br>
$A11 = \sum w_i \cos(\theta _i)^2$ <br>
$A12 = \sum w_i \sin(\theta _i)\cos(\theta _i)$ <br>
$A22 = \sum w_i \sin(\theta _i)^2$ <br>
$B1 = \sum w_i \cos(\theta _i)F$ <br>
$B2 = \sum w_i \sin(\theta _i)F$ <br>

$x = ( A22\times B1 - A12\times B2 )|A|$ <br>
$y = ( -A12\times B1 + A11\times B2 )|A|$ <br>

$P = (x, y)^T = -A^{-1}B$ <br>
$A^{-1} = 	\frac{1}{|A|}\begin{pmatrix}
   A22 & -A12 \\
   -A12 & A11 
\end{pmatrix}$ <br>
$|A| = A11\times A22 - A12^2$ <br>

## 実装内容
$A = \sum \sin(Hdg)^2$ <br>
$B = \sum -\sin(Hdg)\cos(Hdg)$ <br>
$C = \sum \cos(Hdg)^2$ <br>
$D = \sum(\sin(Hdg_i)\cos(Hdg_i)VogE_i - \sin(Hdg_i)^2VogN_i)$ <br>
$E = \sum(\sin(Hdg_i)\cos(Hdg_i)VogN_i - \cos(Hdg_i)^2VogE_i)$ <br>
$F = \sum(VogN_i\sin(Hdg_i)-VogE_i\cos(Hdg_i))^2$ <br>

$mmsiSumValue = \left\{value_{mmsi_0}, value_{mmsi_1}, ... , value_{mmsi_n}\right\}$ <br>
$value_{mmsi_j} = \left\{C, -B, A, D, E, F\right\}$ <br>
$F1 = -(D\cos(\theta _1) + E\sin(\theta _1))/\lambda _1$ <br>
$F2 = -(D\cos(\theta _2) + E\sin(\theta _2))/\lambda _2$ <br>

## 固有値の計算
$\psi _1 = \frac{1}{2} \arctan (\sum w_i \sin 2\theta_i / \sum w_i \cos 2\theta_i)$ <br>
$\psi _2 = \psi _1 + \pi/2$ <br>
$\lambda_{1,2} = \frac{1}{2}(\sum w_i \mp \sqrt((\sum w_i \sin 2\theta_i)^2+(\sum w_i \cos 2\theta_i)^2))$ <br>