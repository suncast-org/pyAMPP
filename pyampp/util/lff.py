import numpy as np
import warnings

__author__     = "Alexey G. Stupishin"
__email__      = "agstup@yandex.ru"
__copyright__  = "SUNCAST project, 2024"
__credits__    = ["Sergey A. Anfinogentov"]
__license__    = "MIT"
__version__    = "0.9.1"
__maintainer__ = "Alexey G. Stupishin"
__status__     = "beta"

class mf_lfff:
    def __init__(self):
        pass

    def set_field(self, field2D, pad = (1, 1)):
        # prepare
        size = np.shape(field2D)
        pad_size = np.ceil(size * (1 + np.array(pad, dtype = np.float64, order = 'C'))).astype(int)
        pad_size_half = (pad_size+1) // 2
        pad_size = pad_size_half * 2
        field_pad = np.zeros(pad_size, dtype = np.float64, order = 'C')
        field_pad[:size[0], :size[1]] = field2D
        self.__field_av = np.mean(field_pad)
        field_pad -= self.__field_av
    
        # FFT2
        self.__field_fft = np.fft.fft2(field_pad)
    
        # uv-domain coefficients
        u_vect = np.concatenate((np.linspace(0, pad_size_half[0], num = pad_size_half[0], endpoint = False), -np.linspace(pad_size_half[0], 0, num = pad_size_half[0], endpoint = False))) / pad_size[0]
        self.__u = np.tile(u_vect, (pad_size[1], 1)).transpose()
        v_vect = np.concatenate((np.linspace(0, pad_size_half[1], num = pad_size_half[1], endpoint = False), -np.linspace(pad_size_half[1], 0, num = pad_size_half[1], endpoint = False))) / pad_size[1]
        self.__v = np.tile(v_vect, (pad_size[0], 1))

        self.__q = np.sqrt(self.__u**2 + self.__v**2)

        self.__size = size

        pass

    def lfff_at_z(self, z, alpha = 0, directive_cosines = (0, 0, 1)):
        
        if alpha == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                invqv = 1 / (self.__u*directive_cosines[0] + self.__v*directive_cosines[1] + 1j*self.__q*directive_cosines[2])
            invqv[np.isinf(invqv) | np.isnan(invqv)] = 0

            G = self.__field_fft * invqv * np.exp(-2*np.pi*self.__q*z)

            bx = (np.fft.ifft2(self.__u * G)).real
            by = (np.fft.ifft2(self.__v * G)).real
            bz = (np.fft.ifft2(1j*self.__q * G)).real + self.__field_av / directive_cosines[2]
            pass
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                k = np.sqrt(4*(np.pi*self.__q)**2 - alpha**2);
            k[k.imag != 0 | np.isinf(k) | np.isnan(k)] = 0
    
            ukva = self.__u*k - self.__v*alpha
            vkua = self.__v*k + self.__u*alpha
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                invqv = 1 / (-1j*(directive_cosines[0]*ukva + directive_cosines[1]*vkua)/(2*np.pi*self.__q**2) + directive_cosines[2])
            invqv[np.isinf(invqv) | np.isnan(invqv)] = 0

            G = self.__field_fft * invqv * np.exp(-k*z)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                indq = 1 / (2*np.pi*self.__q**2);
            indq[np.isinf(indq) | np.isnan(indq)] = 0
    
            bx = (np.fft.ifft2(-1j*ukva*indq * G)).real
            by = (np.fft.ifft2(-1j*vkua*indq * G)).real
            bz = (np.fft.ifft2(G) + self.__field_av / directive_cosines[2]).real
            pass

        return dict(bx = bx[:self.__size[0], :self.__size[1]]
                  , by = by[:self.__size[0], :self.__size[1]]
                  , bz = bz[:self.__size[0], :self.__size[1]]
                   )

    def lfff_cube(self, nz, alpha = 0, directive_cosines = (0, 0, 1)):
        bx = np.zeros((self.__size[0], self.__size[1], nz))        
        by = np.zeros((self.__size[0], self.__size[1], nz))        
        bz = np.zeros((self.__size[0], self.__size[1], nz))        

        for k in range(0, nz):
            res = self.lfff_at_z(k, alpha, directive_cosines)
            bx[:,:,k] = res['bx']
            by[:,:,k] = res['by']
            bz[:,:,k] = res['bz']
            pass

        return dict(bx = bx
                  , by = by
                  , bz = bz
                   )