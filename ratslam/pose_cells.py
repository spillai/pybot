# =============================================================================
# Federal University of Rio Grande do Sul (UFRGS)
# Connectionist Artificial Intelligence Laboratory (LIAC)
# Renato de Pontes Pereira - rppereira@inf.ufrgs.br
# =============================================================================
# Copyright (c) 2013 Renato de Pontes Pereira, renato.ppontes at gmail dot com
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
import itertools
from ._globals import *

class PoseCells(object):
    '''Pose Cell module.'''

    def __init__(self):
        '''Initializes the Pose Cell module.'''

        self.cells = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH])
        self.active = a, b, c = [PC_DIM_XY/2, PC_DIM_XY/2, PC_DIM_TH/2]
        self.cells[a, b, c] = 1

    def compute_activity_matrix(self, xywrap, thwrap, wdim, pcw): 
        '''Compute the activation of pose cells.'''
        
        # The goal is to return an update matrix that can be added/subtracted
        # from the posecell matrix
        pca_new = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH])
        
        # for nonzero posecell values  
        indices = np.nonzero(self.cells)

        for i,j,k in itertools.izip(*indices):
            pca_new[np.ix_(xywrap[i:i+wdim], 
                           xywrap[j:j+wdim],
                           thwrap[k:k+wdim])] += self.cells[i,j,k]*pcw
         
        return pca_new


    def get_pc_max(self, xywrap, thwrap):
        '''Find the x, y, th center of the activity in the network.'''
        
        x, y, z = np.unravel_index(np.argmax(self.cells), self.cells.shape)
        
        z_posecells = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH]) 
      
        zval = self.cells[np.ix_(
            xywrap[x:x+PC_CELLS_TO_AVG*2], 
            xywrap[y:y+PC_CELLS_TO_AVG*2], 
            thwrap[z:z+PC_CELLS_TO_AVG*2]
        )]
        z_posecells[np.ix_(
            PC_AVG_XY_WRAP[x:x+PC_CELLS_TO_AVG*2], 
            PC_AVG_XY_WRAP[y:y+PC_CELLS_TO_AVG*2], 
            PC_AVG_TH_WRAP[z:z+PC_CELLS_TO_AVG*2]
        )] = zval
        
        # get the sums for each axis
        x_sums = np.sum(np.sum(z_posecells, 2), 1) 
        y_sums = np.sum(np.sum(z_posecells, 2), 0)
        th_sums = np.sum(np.sum(z_posecells, 1), 0)
        th_sums = th_sums[:]
        
        # now find the (x, y, th) using population vector decoding to handle 
        # the wrap around 
        x = (np.arctan2(np.sum(PC_XY_SUM_SIN_LOOKUP*x_sums), 
                        np.sum(PC_XY_SUM_COS_LOOKUP*x_sums)) * \
            PC_DIM_XY/(2*np.pi)) % (PC_DIM_XY)
            
        y = (np.arctan2(np.sum(PC_XY_SUM_SIN_LOOKUP*y_sums), 
                        np.sum(PC_XY_SUM_COS_LOOKUP*y_sums)) * \
            PC_DIM_XY/(2*np.pi)) % (PC_DIM_XY)
            
        th = (np.arctan2(np.sum(PC_TH_SUM_SIN_LOOKUP*th_sums), 
                         np.sum(PC_TH_SUM_COS_LOOKUP*th_sums)) * \
             PC_DIM_TH/(2*np.pi)) % (PC_DIM_TH)

        # print x, y, th
        return (x, y, th)

    def __call__(self, view_cell, vtrans, vrot):
        '''Execute an interation of pose cells.

        :param view_cell: the last most activated view cell.
        :param vtrans: the translation of the robot given by odometry.
        :param vrot: the rotation of the robot given by odometry.
        :return: a 3D-tuple with the (x, y, th) index of most active pose cell.
        '''
        vtrans = vtrans*POSECELL_VTRANS_SCALING

        # if this isn't a new vt then add the energy at its associated posecell
        # location
        if not view_cell.first:
            act_x = np.min([np.max([int(np.floor(view_cell.x_pc)), 1]), PC_DIM_XY]) 
            act_y = np.min([np.max([int(np.floor(view_cell.y_pc)), 1]), PC_DIM_XY])
            act_th = np.min([np.max([int(np.floor(view_cell.th_pc)), 1]), PC_DIM_TH])

            # print [act_x, act_y, act_th]
        # this decays the amount of energy that is injected at the vt's
        # posecell location
        # this is important as the posecell Posecells will errounously snap 
        # for bad vt matches that occur over long periods (eg a bad matches that
        # occur while the agent is stationary). This means that multiple vt's
        # need to be recognised for a snap to happen
            energy = PC_VT_INJECT_ENERGY*(1./30.)*(30 - np.exp(1.2 * view_cell.decay))
            if energy > 0:
                self.cells[act_x, act_y, act_th] += energy
        #===============================


        # local excitation - PC_le = PC elements * PC weights
        self.cells = self.compute_activity_matrix(PC_E_XY_WRAP, 
                                                  PC_E_TH_WRAP, 
                                                  PC_W_E_DIM, 
                                                  PC_W_EXCITE)
        # print np.max(self.cells)
        # raw_input()

        # local inhibition - PC_li = PC_le - PC_le elements * PC weights
        self.cells = self.cells-self.compute_activity_matrix(PC_I_XY_WRAP, 
                                                             PC_I_TH_WRAP, 
                                                             PC_W_I_DIM, 
                                                             PC_W_INHIB) 


        # local global inhibition - PC_gi = PC_li elements - inhibition
        self.cells[self.cells < PC_GLOBAL_INHIB] = 0
        self.cells[self.cells >= PC_GLOBAL_INHIB] -= PC_GLOBAL_INHIB

        # normalization
        total = np.sum(self.cells)
        self.cells = self.cells/total


        # Path Integration
        # vtrans affects xy direction
        # shift in each th given by the th
        for dir_pc in xrange(PC_DIM_TH): 
            direction = np.float64(dir_pc-1) * PC_C_SIZE_TH
            # N,E,S,W are straightforward
            if (direction == 0):
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc] * (1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], 1, 1)*vtrans

            elif direction == np.pi/2:
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], 1, 0)*vtrans

            elif direction == np.pi:
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], -1, 1)*vtrans

            elif direction == 3*np.pi/2:
                self.cells[:,:,dir_pc] = \
                    self.cells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.cells[:,:,dir_pc], -1, 0)*vtrans

            else:
                pca90 = np.rot90(self.cells[:,:,dir_pc], 
                              int(np.floor(direction *2/np.pi)))
                dir90 = direction - int(np.floor(direction*2/np.pi)) * np.pi/2


                # extend the Posecells one unit in each direction (max supported at the moment)
                # work out the weight contribution to the NE cell from the SW, NW, SE cells 
                # given vtrans and the direction
                # weight_sw = v * cos(th) * v * sin(th)
                # weight_se = (1 - v * cos(th)) * v * sin(th)
                # weight_nw = (1 - v * sin(th)) * v * sin(th)
                # weight_ne = 1 - weight_sw - weight_se - weight_nw
                # think in terms of NE divided into 4 rectangles with the sides
                # given by vtrans and the angle
                pca_new = np.zeros([PC_DIM_XY+2, PC_DIM_XY+2])   
                pca_new[1:-1, 1:-1] = pca90 
                
                weight_sw = (vtrans**2) *np.cos(dir90) * np.sin(dir90)
                weight_se = vtrans*np.sin(dir90) - \
                            (vtrans**2) * np.cos(dir90) * np.sin(dir90)
                weight_nw = vtrans*np.cos(dir90) - \
                            (vtrans**2) *np.cos(dir90) * np.sin(dir90)
                weight_ne = 1.0 - weight_sw - weight_se - weight_nw
          
                pca_new = pca_new*weight_ne + \
                          np.roll(pca_new, 1, 1) * weight_nw + \
                          np.roll(pca_new, 1, 0) * weight_se + \
                          np.roll(np.roll(pca_new, 1, 1), 1, 0) * weight_sw

                pca90 = pca_new[1:-1, 1:-1]
                pca90[1:, 0] = pca90[1:, 0] + pca_new[2:-1, -1]
                pca90[1, 1:] = pca90[1, 1:] + pca_new[-1, 2:-1]
                pca90[0, 0] = pca90[0, 0] + pca_new[-1, -1]

                #unrotate the pose cell xy layer
                self.cells[:,:,dir_pc] = np.rot90(pca90, 
                                                   4 - int(np.floor(direction * 2/np.pi)))


        # Path Integration - Theta
        # Shift the pose cells +/- theta given by vrot
        if vrot != 0: 
            weight = (np.abs(vrot)/PC_C_SIZE_TH)%1
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(vrot) * int(np.floor(abs(vrot)/PC_C_SIZE_TH)))
            shift2 = int(np.sign(vrot) * int(np.ceil(abs(vrot)/PC_C_SIZE_TH)))
            self.cells = np.roll(self.cells, shift1, 2) * (1.0 - weight) + \
                             np.roll(self.cells, shift2, 2) * (weight)
        
        self.active = self.get_pc_max(PC_AVG_XY_WRAP, PC_AVG_TH_WRAP)
        return self.active
    
