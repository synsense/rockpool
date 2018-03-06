function PlotEigenspectrum(vfEigs)

newplot;
plot(complex(vfEigs), '.');

bHold = ishold;
hold on;

axis equal;
xlabel('Re');
ylabel('Im');

vfTheta = linspace(0, 2*pi, 100);

plot(exp(-1i * vfTheta), 'r--');
plot([0 0], ylim, 'r--');
plot([1 1], ylim, 'r--');

if ~bHold
   hold off;
end

