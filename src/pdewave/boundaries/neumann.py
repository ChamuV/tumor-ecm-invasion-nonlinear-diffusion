# src/pdewave/boundaries/neumann.py


class NeumannBC1D:
    name = "neumann"

    @staticmethod
    def apply(u):
        u[0] = u[1]
        u[-1] = u[-2]
        return u

    @staticmethod
    def apply_to_matrix(L):
        L[0, 1] = 2.0
        L[-1, -2] = 2.0
        return L

    @staticmethod
    def apply_to_variable_diffusion_diagonals(a, lower, center, upper):
        a_right = 0.5 * (a[0] + a[1])
        center[0] = -2.0 * a_right
        upper[0] = 2.0 * a_right

        a_left = 0.5 * (a[-2] + a[-1])
        center[-1] = -2.0 * a_left
        lower[-1] = 2.0 * a_left

        return lower, center, upper
    
    @staticmethod

    def apply_to_ecm_diffusion_diagonals(m, coefficient, lower, center, upper):
        m_right = 0.5 * (m[0] + m[1])
        D_right = coefficient(m_right)
        center[0] = -2.0 * D_right
        upper[0] = 2.0 * D_right
        
        m_left = 0.5 * (m[-2] + m[-1])
        D_left = coefficient(m_left)
        center[-1] = -2.0 * D_left
        lower[-1] = 2.0 * D_left

        return lower, center, upper