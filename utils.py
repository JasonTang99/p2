from collections import namedtuple
from argparse import ArgumentParser

# Export the namedtuple
Args = namedtuple('Args', [
    'hidden', 'nz', 'ngf', 'nc', 'epsilon', 'delta', 'noise_multiplier', 
    'c_p', 'lr', 'beta1', 'batch_size', 'n_d', 'n_g', 'activation', 'lambda_gp'
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
    parser.add_argument("--activation", type=str, default="LeakyReLU")
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    args = parser.parse_args()

    # Convert args to namedtuple
    args = Args(
        hidden=args.hidden, nz=args.nz, ngf=args.ngf, nc=args.nc, 
        epsilon=args.epsilon, delta=args.delta, noise_multiplier=args.noise_multiplier, 
        c_p=args.c_p, lr=args.lr, beta1=args.beta1, batch_size=args.batch_size, 
        n_d=args.n_d, n_g=args.n_g, activation=args.activation, lambda_gp=args.lambda_gp
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

# Inverse of generate_run_id
def parse_run_id(run_id):
    # Strip any extra prefixs
    for prefix in ["private_", "public_", "ae-enc_", "ae-grad_", "wgan_"]:
        if run_id.startswith(prefix):
            run_id = run_id.split(prefix, 1)[1]
            break
    
    # Parse run id to args (named tuple)
    args = run_id.split("_")
    # Convert args[0] to list
    args[0] = [int(a) for a in args[0].split("-")]
    
    float_args = [args[0]]
    for a in args[1:]:
        try:
            if "." in a or "e" in a:
                float_args.append(float(a))
            else:
                float_args.append(int(a))
        except ValueError:
            float_args.append(a)
    args = float_args

    args = Args(
        hidden=args[0], nz=args[1], ngf=args[2], nc=args[3],
        epsilon=args[4], delta=args[5], noise_multiplier=args[6],
        c_p=args[7], lr=args[8], beta1=args[9], batch_size=args[10],
        n_d=args[11], n_g=args[12], activation=args[13], lambda_gp=args[14]
    )
    return args

if __name__ == "__main__":
    # python utils.py --hidden 64 16 --nz 100 --ngf 32 --nc 1 --epsilon 50.0 --delta 1e-6 --noise_multiplier 0.1 --c_p 0.01 --lr 1e-3 --beta1 0.5 --batch_size 32 --n_d 3 --n_g 10000 --activation leaky_relu --lambda_gp 10.0

    # Test get_input_args and generate_run_id
    args = get_input_args()
    run_id = generate_run_id(args)
    print(run_id)

    # Test parse_run_id
    # run_id = "128_100_32_1_50.0_1e-06_0.1_0.01_0.001_0.5_32_3_10000_leaky_relu_10.0"
    args = parse_run_id(run_id)
    print(args)
    print(args.delta * 100)
