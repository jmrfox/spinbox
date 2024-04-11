from quap import *



class ExactGFMC:
    def __init__(self,n_particles):
        self.n_particles = n_particles
        self.ident = GFMCSpinIsospinOperator(n_particles)
        self.sig = [[GFMCSpinIsospinOperator(n_particles).sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
        self.tau = [[GFMCSpinIsospinOperator(n_particles).tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]

    def g_pade_sig(self, dt, asig, i, j):
        out = GFMCSpinIsospinOperator(self.n_particles).zeros()
        for a in range(3):
            for b in range(3):
                out += asig[a, i, b, j] * self.sig[i][a] * self.sig[j][b]
        out = -0.5 * dt * out
        return out.exponentiate()


    def g_pade_sigtau(self, dt, asigtau, i, j):
        out = GFMCSpinIsospinOperator(self.n_particles).zeros()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    out += asigtau[a, i, b, j] * self.sig[i][a] * self.sig[j][b] * self.tau[i][c] * self.tau[j][c]
        out = -0.5 * dt * out
        return out.exponentiate()


    def g_pade_tau(self, dt, atau, i, j):
        out = GFMCSpinIsospinOperator(self.n_particles).zeros()
        for c in range(3):
            out += atau[i, j] * self.tau[i][c] * self.tau[j][c]
        out = -0.5 * dt * out
        return out.exponentiate()


    def g_pade_coul(self, dt, v, i, j):
        out = self.ident + self.tau[i][2] + self.tau[j][2] + self.tau[i][2] * self.tau[j][2]
        out = -0.125 * v[i, j] * dt * out
        return out.exponentiate()


    def g_coulomb_onebody(self, dt, v, i):
        """just the one-body part of the expanded coulomb propagator
        for use along with auxiliary field propagators"""
        out = - 0.125 * v * dt * self.tau[i][2]
        return out.exponentiate()


    def g_ls_linear(self, gls, i):
        # linear approx to LS
        out = GFMCSpinIsospinOperator(self.n_particles)
        for a in range(3):
            out = (self.ident - 1.j * gls[a, i] * self.sig[i][a]) * out 
        return out

    def g_ls_onebody(self, gls_ai, i, a):
        # one-body part of the LS propagator factorization
        out = - 1.j * gls_ai * self.sig[i][a]
        return out.exponentiate()

    def g_ls_twobody(self, gls_ai, gls_bj, i, j, a, b):
        # two-body part of the LS propagator factorization
        out = 0.5 * gls_ai * gls_bj * self.sig[i][a] * self.sig[j][b]
        return out.exponentiate()

    def make_g_exact(self, dt, pot, controls):
        # compute exact bracket
        g_exact = self.ident.copy()
        pairs_ij = interaction_indices(self.n_particles)
        for i,j in pairs_ij:
            if controls['sigma']:
                g_exact = self.g_pade_sig(dt, pot.sigma, i, j) * g_exact
            if controls['sigmatau']:
                g_exact = self.g_pade_sigtau(dt, pot.sigmatau, i, j) * g_exact 
            if controls['tau']:
                g_exact = self.g_pade_tau(dt, pot.tau, i, j) * g_exact
            if controls['coulomb']:
                g_exact = self.g_pade_coul(dt, pot.coulomb, i, j) * g_exact
        #  LS
        if controls['spinorbit']:
            for i in range(self.n_particles):
                g_exact = self.g_ls_linear(pot.spinorbit, i) * g_exact
            # for i in range(n_particles):
            #     for a in range(3):
            #         g_exact = g_ls_onebody(gls[a, i], i, a) * g_exact
            # for i in range(n_particles):
            #     for j in range(n_particles):
            #         for a in range(3):
            #             for b in range(3):
            #                 g_exact = g_ls_twobody(gls[a, i], gls[b, j], i, j, a, b) * g_exact
        return g_exact

