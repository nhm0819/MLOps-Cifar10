def helloworld(a):
    msg = f"Hello World! {a}!"
    print(msg)
    return msg


if __name__ == "__main__":  # pragma: no cover
    helloworld("Hong")
    assert helloworld("Hongmin")
