successes = []
failures = []

with open("bringup_results.txt", "r") as f:
    for line in f:
        if line.startswith("SUCCESS:"):
            name = line.strip().split("SUCCESS:")[1].strip()
            successes.append(name)
        if line.startswith("FAILURE:"):
            name = line.strip().split("FAILURE:")[1].strip()
            failures.append(name)

print(f"Total successful: {len(successes)}")
print("Successful programs:")
for name in successes:
    print(name)

print("-------------")
print(f"Total failures: {len(failures)}")
print("Unsuccessful programs:")
for name in failures:
    print(name)