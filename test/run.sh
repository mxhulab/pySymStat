# Compute optimal solutions.
for space in SO3 S2; do
    for group in C2 C7 D2 D7 T O I; do
        python compute_opt.py ${space} ${group}
    done
done

# Compute optimal solutions for approximate problems.
for space in SO3 S2; do
    for type in arithmetic geometric; do
        for group in C2 C7 D2 D7 T O I; do
            python compute_app.py ${space} ${type} ${group}
        done
    done
done

# Compute nug solutions.
for space in SO3 S2; do
    for type in arithmetic geometric; do
        for group in C2 C7 D2 D7 T O I; do
            python compute_nug.py ${space} ${type} ${group}
        done
    done
done

# Compute nug solutions with other hyper-parameters.
for group in C2 C7 D2 D7 T O I; do
    python compute_nug.py SO3 arithmetic ${group} --capacity 4
    python compute_nug.py SO3 arithmetic ${group} --capacity 12
    python compute_nug.py SO3 arithmetic ${group} --threshold 0
    python compute_nug.py SO3 arithmetic ${group} --threshold 0.5
done

# Write a summary report.
python summary.py
