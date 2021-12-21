import argparse
import csv
import json
import pathlib
import time

import ecole as ec
import numpy as np


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ec.observation.Pseudocosts()
        self.strong_branching_function = ec.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="start")
    parser.add_argument("--end", type=int, default=-1, help="end")
    parser.add_argument("--exp_name", type=str, default="", help="end")
    parser.add_argument(
        "task", help="Task to evaluate.", choices=["primal", "dual", "config"],
    )
    parser.add_argument(
        "problem",
        help="Problem benchmark to process.",
        choices=["item_placement", "load_balancing", "anonymous"],
    )
    parser.add_argument(
        "-t",
        "--timelimit",
        help="Episode time limit (in seconds).",
        default=argparse.SUPPRESS,
        type=float,
    )
    parser.add_argument(
        "-d", "--debug", help="Print debug traces.", action="store_true",
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Instance folder to evaluate.",
        default="valid",
        type=str,
        choices=("valid", "test", "train"),
    )
    args = parser.parse_args()

    print(f"Evaluating the {args.task} task agent.")

    # collect the instance files
    if args.problem == "item_placement":
        instances_path = pathlib.Path(
            f"../../instances/1_item_placement/{args.folder}/"
        )
        instances_path = pathlib.Path(
            f"/data/ml4co-competition/instances/1_item_placement/{args.folder}/"
        )
        results_file = pathlib.Path(
            f"results/{args.task}/1_item_placement{args.exp_name}{args.start}_{args.end}.csv"
        )
    elif args.problem == "load_balancing":
        instances_path = pathlib.Path(
            f"../../instances/2_load_balancing/{args.folder}/"
        )
        instances_path = pathlib.Path(
            f"/data/ml4co-competition/instances/2_load_balancing/{args.folder}/"
        )
        results_file = pathlib.Path(
            f"results/{args.task}/2_load_balancing{args.exp_name}{args.start}_{args.end}.csv"
        )
    elif args.problem == "anonymous":
        instances_path = pathlib.Path(f"../../instances/3_anonymous/{args.folder}/")
        instances_path = pathlib.Path(
            f"/data/ml4co-competition/instances/3_anonymous/{args.folder}/"
        )
        results_file = pathlib.Path(
            f"results/{args.task}/3_anonymous{args.exp_name}{args.start}_{args.end}.csv"
        )

    print(f"Processing instances from {instances_path.resolve()}")
    instance_files = list(instances_path.glob("*.mps.gz"))[args.start : args.end]

    print(f"Saving results to {results_file.resolve()}")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_fieldnames = [
        "instance",
        "seed",
        "initial_primal_bound",
        "initial_dual_bound",
        "objective_offset",
        "cumulated_reward",
    ]
    with open(results_file, mode="w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
        writer.writeheader()

    import sys

    sys.path.insert(1, str(pathlib.Path.cwd()))

    # set up the proper agent, environment and goal for the task
    if args.task == "primal":
        from agents.primal import Policy, ObservationFunction
        from environments import RootPrimalSearch as Environment
        from rewards import TimeLimitPrimalIntegral as BoundIntegral

        time_limit = 5 * 60

    elif args.task == "dual":
        from agents.dual import (
            Policy,
            ObservationFunction,
        )  # agents.dual submissions.random.
        from environments import Branching as Environment  # environments
        from rewards import TimeLimitDualIntegral as BoundIntegral  # rewards

        time_limit = 15 * 60

    elif args.task == "config":
        from agents.config import Policy, ObservationFunction
        from environments import Configuring as Environment
        from rewards import TimeLimitPrimalDualIntegral as BoundIntegral

        time_limit = 15 * 60

    # override from command-line argument if provided
    time_limit = getattr(args, "timelimit", time_limit)

    policy = Policy(problem=args.problem)
    observation_function = ObservationFunction(problem=args.problem)

    integral_function = BoundIntegral()

    env = Environment(  # Environment
        time_limit=time_limit,
        observation_function=observation_function,  # (ExploreThenStrongBranch(expert_probability=1.0),observation_function),              #observation_function, ec.observation.NodeBipartite()
        reward_function=-integral_function,  # negated integral (minimization)
    )

    # evaluation loop
    instance_count = 0
    for seed, instance in enumerate(instance_files):
        instance_count += 1
        # seed both the agent and the environment (deterministic behavior)
        observation_function.seed(seed)
        policy.seed(seed)
        env.seed(seed)

        # read the instance's initial primal and dual bounds from JSON file
        with open(instance.with_name(instance.stem).with_suffix(".json")) as f:
            instance_info = json.load(f)

        # set up the reward function parameters for that instance
        initial_primal_bound = instance_info["primal_bound"]
        initial_dual_bound = instance_info["dual_bound"]
        objective_offset = 0

        integral_function.set_parameters(
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
            objective_offset=objective_offset,
        )

        print()
        print(f"Instance {instance.name}")
        print(f"  seed: {seed}")
        print(f"  initial primal bound: {initial_primal_bound}")
        print(f"  initial dual bound: {initial_dual_bound}")
        print(f"  objective offset: {objective_offset}")

        # reset the environment
        observation, action_set, reward, done, info = env.reset(
            str(instance), objective_limit=initial_primal_bound
        )

        if args.debug:
            print(f"  info: {info}")
            print(f"  reward: {reward}")
            print(f"  action_set: {action_set}")

        cumulated_reward = 0  # discard initial reward

        cumulated_rewards = []
        # loop over the environment
        while not done:
            action = policy(action_set, observation)
            # (scores, scores_are_expert), node_observation = observation
            # action = action_set[scores[action_set].argmax()]
            observation, action_set, reward, done, info = env.step(action)
            if args.debug:
                print(f"  action: {action}")
                print(f"  info: {info}")
                print(f"  reward: {reward}")
                print(f"  action_set: {action_set}")

            cumulated_reward += reward

            cumulated_rewards.append(cumulated_reward)

        print(f"  cumulated reward (to be maximized): {cumulated_reward}")
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

        np.save(f"/data/load_random_{instance_count}.npy", cumulated_rewards)
        # print(step_count)
        # save instance results
        with open(results_file, mode="a") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
            writer.writerow(
                {
                    "instance": str(instance),
                    "seed": seed,
                    "initial_primal_bound": initial_primal_bound,
                    "initial_dual_bound": initial_dual_bound,
                    "objective_offset": objective_offset,
                    "cumulated_reward": cumulated_reward,
                }
            )
