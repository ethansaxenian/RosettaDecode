use Proc::Fork;
# parent code ...
run_fork {
    child {
        # child code ...
    }
};
# parent code continues ...