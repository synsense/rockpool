function [mfWHat, mfJHat] = DetectUnstablePartitions(mfW, vnInh, fInhTauFactor)

nNetSize = size(mfW, 1);

% - Compute Jacobian
mfJ = mfW - eye(size(mfW));
mfJ(vnInh, :) = mfJ(vnInh, :) ./ fInhTauFactor;

% - Check for unstable eigenvectors
[V, D] = eig(mfJ);

% - Examine eigenvectors with positive real part
vnPREigs = find(real(diag(D)) > 0)';
vbWipeEig = false(nNetSize, 1);
for nEig = vnPREigs
   vfEigVec = V(:, nEig);
   vbDestPartition = real(vfEigVec) > 0;
   mfWPart = mfW(vbDestPartition, vbDestPartition);
   fMaxEig = eigs(mfWPart, 1, 'lr');
   if (fMaxEig > 0)
      vbWipeEig(nEig) = true;
   end
end

% - Wipe unstable eigenvectors
D(vbWipeEig, vbWipeEig) = 0;
disp(nnz(vbWipeEig));

% - Reconstruct weight matrix
mfJHat = real(V * D / V);
mfWHat = mfJHat;
mfWHat(vnInh, :) = mfJHat(vnInh, :) .* fInhTauFactor;
mfWHat = mfWHat + eye(size(mfWHat));
% mfWHat = mfWHat .* fInhTauFactor * 30;

