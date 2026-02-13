import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import time
from torchvision import datasets, transforms

def load_mnist():
    # Normalize inputs requirement 
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def preprocess(ds):
        x = jnp.array(ds.data.reshape(-1, 784) / 255.0) # Normalization 
        y = jax.nn.one_hot(jnp.array(ds.targets), 10) # encoding 
        return x, y

    x_train_full, y_train_full = preprocess(train_ds)
    x_test, y_test = preprocess(test_ds)
    
    # Appropriate split: 50k train, 10k validation 
    return (x_train_full[:50000], y_train_full[:50000]), (x_train_full[50000:], y_train_full[50000:]), (x_test, y_test)

def init_params(layers, key):
    params = []
    keys = random.split(key, len(layers) - 1)
    for i, (in_d, out_d) in enumerate(zip(layers[:-1], layers[1:])):
        w_key, b_key = random.split(keys[i])
        params.append({"w": random.normal(w_key, (in_d, out_d)) * jnp.sqrt(2/in_d), "b": jnp.zeros((out_d,))})
    return params

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(jnp.dot(x, layer['w']) + layer['b'])
    return jnp.dot(x, params[-1]['w']) + params[-1]['b']

@jit # jit for optimization 
def accuracy_fn(params, x, y):
    logits = vmap(forward, in_axes=(None, 0))(params, x)
    return jnp.mean(jnp.argmax(logits, axis=1) == jnp.argmax(y, axis=1)) * 100

def loss_fn(params, x, y):
    logits = vmap(forward, in_axes=(None, 0))(params, x)
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=1))

@jit # Gradient updates using jax.grad 
def update(params, x, y, lr=0.01):
    grads = grad(loss_fn)(params, x, y)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

def benchmark(batch_size, train, val, test):
    print(f"\n[JAX] Batch Size: {batch_size}")
    params = init_params([784, 512, 512, 10], random.PRNGKey(42))
    for epoch in range(5):
        start = time.time()
        params = update(params, train[0][:batch_size], train[1][:batch_size])
        jax.block_until_ready(params) # Accurate GPU timing 
        dt = time.time() - start
        
        t_acc = accuracy_fn(params, train[0][:batch_size], train[1][:batch_size])
        v_acc = accuracy_fn(params, val[0], val[1])
        loss = loss_fn(params, train[0][:batch_size], train[1][:batch_size])
        
        label = "First Epoch (JIT)" if epoch == 0 else f"Steady Epoch {epoch}"
        print(f"{label}: {dt:.6f}s | Loss: {loss:.4f} | Accuracy: {t_acc:.2f}% | Val_Acc: {v_acc:.2f}%")
    
    print(f"Final Test Accuracy: {accuracy_fn(params, test[0], test[1]):.2f}%")

if __name__ == "__main__":
    train, val, test = load_mnist()
    for b in [64, 256, 1024]: # Batch size comparison 
        benchmark(b, train, val, test)