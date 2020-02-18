#!/usr/bin/env python

"""
Test script for diagonalized fusion testing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt

from etddf.covar_intersect import covar_intersect, gen_sim_transform

def main():
    
    mat_a = np.array([[3,0.5,0.1,-0.2],
                        [0.5,5,0.08,0.03],
                        [0.1,0.08,4,-0.35],
                        [-0.2,0.03,-0.35,0.98]])

    mat_b = np.array([[3.3,0.2],
                        [0.2,4.7]])

    # partial state CI

    # eigenbasis partial state CI
    _,mat_a_eig = np.linalg.eig(mat_a)

    # extract elements for mat b
    mat_a_red = mat_a_eig[0:2,0:2]

    # transform full matrix A
    diag_mat_a = np.diag(np.diag(np.dot(np.linalg.inv(mat_a_eig),np.dot(mat_a,mat_a_eig))))
    print('diag mat a')
    print(diag_mat_a)

    # pad matrix B with zeros to get into same space
    mat_b_padded = np.zeros(mat_a.shape)
    mat_b_padded[0:mat_b.shape[0],0:mat_b.shape[1]] = mat_b
    print('pad mat b')
    print(mat_b_padded)

    # transform padded matrix b into matrix a eigenbasis
    diag_mat_b = np.diag(np.diag(np.dot(np.linalg.inv(mat_a_eig),np.dot(mat_b_padded,mat_a_eig))))
    print('diag mat b')
    print(diag_mat_b)

    # transform matrix b with reduced eigen vector matrix
    mat_b_transformed = np.dot(np.linalg.inv(mat_a_eig)[0:2,0:2],np.dot(mat_b,mat_a_red[0:2,0:2]))
    print('mat b transformed')
    print(mat_b_transformed)


    xc,Pc = covar_intersect(np.zeros((mat_a.shape[0],)),np.zeros((mat_a.shape[0],)),diag_mat_a,diag_mat_b)
    print(Pc)

    # transform back
    mat_a_updated = np.dot(mat_a_eig,np.dot(Pc,np.linalg.inv(mat_a_eig)))
    print('mat a post CI')
    print(mat_a_updated)

    # partial state covariance intersection
    xc_partial,Pc_partial = covar_intersect(np.zeros((mat_b.shape[0],)),np.zeros((mat_b.shape[0],)),diag_mat_a[0:2,0:2],mat_b_transformed)
    print(Pc_partial)

    # compute information delta for conditional update
    invD_a = np.linalg.inv(Pc_partial) - np.linalg.inv(diag_mat_a[0:2,0:2])
    print(invD_a)
    # invDd_a = np.dot(inv(Pc),xc) - np.dot(inv(PaTred),xaTred)

    invD_b = np.linalg.inv(Pc_partial) - np.linalg.inv(mat_b_transformed)
    # invDd_b = np.dot(inv(Pc),xc) - np.dot(inv(PbTred),xbTred)

    # conditional gaussian update
    if (mat_a.shape[0]-Pc_partial.shape[0] == 0) or (mat_a.shape[1]-Pc_partial.shape[1] == 0):
        cond_cov_a = invD_a
        # cond_mean_a = invDd_a
    else:
        cond_cov_a_row1 = np.hstack( (invD_a,np.zeros((Pc_partial.shape[0],mat_a.shape[1]-Pc_partial.shape[1]))) )
        cond_cov_a_row2 = np.hstack( (np.zeros((mat_a.shape[0]-Pc_partial.shape[0],Pc_partial.shape[1])),np.zeros((mat_a.shape[0]-Pc_partial.shape[0],mat_a.shape[0]-Pc_partial.shape[0]))) )
        cond_cov_a = np.vstack( (cond_cov_a_row1,cond_cov_a_row2) )
        # cond_mean_a = np.vstack( (invDd_a,np.zeros((PaT.shape[0]-Pc.shape[0],1))) )

    print(cond_cov_a)

    Va = np.linalg.inv(np.linalg.inv(mat_a) + cond_cov_a)
    print('partial state CI updated mat a')
    print(Va)
    # va = np.dot(Va,np.dot(inv(PaT),xaT) + cond_mean_a)

    # do it all again for nav estimate side
    if (mat_b.shape[0]-Pc_partial.shape[0] == 0) or (mat_b.shape[1]-Pc.shape[1] == 0):
        cond_cov_b = invD_b
        # cond_mean_b = invDd_b
    else:
        cond_cov_b_row1 = np.hstack( (invD_b,np.zeros((Pc.shape[0],mat_b.shape[1]-Pc.shape[1]))) )
        cond_cov_b_row2 = np.hstack( (np.zeros((mat_b.shape[0]-Pc.shape[0],Pc.shape[1])),np.zeros((mat_b.shape[0]-Pc.shape[0],mat_b.shape[0]-Pc.shape[0]))) )
        cond_cov_b = np.vstack( (cond_cov_b_row1,cond_cov_b_row2) )
        # cond_mean_b = np.vstack( (invDd_b,np.zeros((PbT.shape[0]-Pc.shape[0],1))) )

    Vb = np.linalg.inv(np.linalg.inv(mat_b) + cond_cov_b)
    print('partial state CI updated mat b')
    print(Vb)
    # vb = np.dot(Vb,np.dot(inv(PbT),np.reshape(xbT,(xbT.shape[0],1))) + cond_mean_b)

    # transform back from eigenbasis
    mat_a_updated_pci = np.dot(mat_a_eig,np.dot(Va,np.linalg.inv(mat_a_eig)))
    print('mat a updated pci')
    print(mat_a_updated_pci)

    mat_b_updated_pci = np.dot(mat_a_eig[0:2,0:2],np.dot(Vb,np.linalg.inv(mat_a_eig)[0:2,0:2]))
    print('mat b updated pci')
    print(mat_b_updated_pci)


    plt.figure()
    plt.grid(True)
    ax = plt.gca()
    plot_cov_ellipse(mat_b,[0,0])
    # plot_cov_ellipse(mat_b_updated_pci,[0,0])

    plt.show()


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = pt.Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip



if __name__ == "__main__":
    main()