import torch
import matplotlib.pyplot as plt

def draw_unitcircle(p=None, m=None):
    theta = torch.linspace(0, 2 * torch.pi, 100)
    x = torch.cos(theta)
    y = torch.sin(theta)
    
    plt.plot(x, y, 'b', linewidth=4)
    plt.plot([-1, 1], [0, 0], 'k')
    plt.plot([0, 0], [-1, 1], 'k')
    plt.title('Inverse poles')
    plt.axis([-1.02, 1.02, -1.02, 1.02])
    plt.gca().set_aspect('equal', adjustable='box')
    
    if p is not None and m is not None:
        mult = 'Pole configuration: a=('
        for i in range(len(p)):
            plt.plot(torch.real(p[i]), torch.imag(p[i]), 'sr', markersize=16, markerfacecolor='g')
            plt.text(torch.real(p[i]), torch.imag(p[i]), f'a_{i}', horizontalalignment='center', fontsize=10)
            mult += f'a_{i},'
        mult = mult[:-1] + ') and m=('
        for i in range(len(p)):
            mult += f'{m[i]},'
        mult = mult[:-1] + ')'
        plt.title(mult)
    
    #plt.show()
    
    return x, y

