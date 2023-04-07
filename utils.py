from collections import namedtuple
from argparse import ArgumentParser

# Export the namedtuple
Args = namedtuple('Args', [
    'hidden', 'nz', 'ngf', 'nc', 'epsilon', 'delta', 'noise_multiplier', 
    'c_p', 'lr', 'beta1', 'batch_size', 'n_d', 'n_g'
])

def get_input_args():
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--hidden", nargs="+", type=int, default=[64, 16])
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--nc", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=50.0)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--noise_multiplier", type=float, default=0.1)
    parser.add_argument("--c_p", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_d", type=int, default=3)
    parser.add_argument("--n_g", type=int, default=int(1e4))
    args = parser.parse_args()

    # Convert args to namedtuple
    args = Args(
        hidden=args.hidden, nz=args.nz, ngf=args.ngf, nc=args.nc, 
        epsilon=args.epsilon, delta=args.delta, noise_multiplier=args.noise_multiplier, 
        c_p=args.c_p, lr=args.lr, beta1=args.beta1, batch_size=args.batch_size, 
        n_d=args.n_d, n_g=args.n_g
    )
    return args

def generate_run_id(args):
    # Generate run id from args (named tuple)
    run_id = ""
    for field, arg in zip(args._fields, args):
        if type(arg) == list:
            arg = "-".join([str(a) for a in arg])
        run_id += f"{arg}_"
    return run_id.rstrip("_")


if __name__ == "__main__":
    # python utils.py --hidden 64 16 --nz 100 --ngf 32 --nc 1 --epsilon 50.0 --delta 1e-6 --noise_multiplier 0.1 --c_p 0.01 --lr 1e-3 --beta1 0.5 --batch_size 32 --n_d 3 --n_g 10000

    # Test get_input_args and generate_run_id
    args = get_input_args()
    print(generate_run_id(args))
