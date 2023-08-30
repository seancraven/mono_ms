from dyna.types import (
    ActorCriticHyperParams,
    DynaHyperParams,
    TransitionModelHyperParams,
)

# PJRL NUM_UPDATES = 976
# PJRL MINIBATCH_SIZE = 128
CARTPOLE_AC_HYP = ActorCriticHyperParams(
    NUM_UPDATES=None,
)
CARTPOLE_TM_HYP = TransitionModelHyperParams()

CARTPOLE_HYP = DynaHyperParams(
    ac_hyp=CARTPOLE_AC_HYP,
    model_hyp=CARTPOLE_TM_HYP,
)


# PJRL NUM_UPDATES = 500
CATCH_AC_HYP = ActorCriticHyperParams(
    NUM_UPDATES=None, MINIBATCH_SIZE=10, PRIV_NUM_TIMESTEPS=10
)

CATCH_TM_HYP = TransitionModelHyperParams()
CATCH_HYP = DynaHyperParams(
    ac_hyp=CATCH_AC_HYP,
    model_hyp=CATCH_TM_HYP,
)


def make_hyp_cp(
    num_dyna_itr: int = 976, total_mdp_timesteps: int = 500_000
) -> DynaHyperParams:
    cp_hyp = CARTPOLE_HYP._replace(NUM_UPDATES=num_dyna_itr)
    num_ac_upd = total_mdp_timesteps // (
        cp_hyp.ac_hyp.MINIBATCH_SIZE * cp_hyp.NUM_ENVS * cp_hyp.NUM_UPDATES
    )
    new_ac = CARTPOLE_AC_HYP._replace(NUM_UPDATES=num_ac_upd)
    cp_hyp = cp_hyp._replace(ac_hyp=new_ac)
    total_updates = (
        cp_hyp.NUM_UPDATES
        * cp_hyp.ac_hyp.NUM_UPDATES
        * cp_hyp.ac_hyp.MINIBATCH_SIZE
        * cp_hyp.NUM_ENVS
    )
    print("Total_updates", total_updates)
    return cp_hyp


def make_hyp_catch(
    num_dyna_itr: int = 500,
    total_mdp_timesteps: int = 20_000,
) -> DynaHyperParams:
    cp_hyp = CATCH_HYP._replace(NUM_UPDATES=num_dyna_itr)
    num_ac_upd = total_mdp_timesteps // (
        cp_hyp.ac_hyp.MINIBATCH_SIZE * cp_hyp.NUM_ENVS * cp_hyp.NUM_UPDATES
    )
    new_ac = CATCH_AC_HYP._replace(NUM_UPDATES=num_ac_upd)
    cp_hyp = cp_hyp._replace(ac_hyp=new_ac)
    total_updates = (
        cp_hyp.NUM_UPDATES
        * cp_hyp.ac_hyp.NUM_UPDATES
        * cp_hyp.ac_hyp.MINIBATCH_SIZE
        * cp_hyp.NUM_ENVS
    )
    print("Total_updates", total_updates)
    return cp_hyp
