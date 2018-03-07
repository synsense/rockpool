function mfWHat = WipeRealEigs(mfW)

[V, D] = eig(mfW);

vbWipeEig = imag(diag(D)) < eps;

D(vbWipeEig, vbWipeEig) = 0;

mfWHat = real(V * D / V);

