println("Parent running.")
@async(begin sleep(1); println("This is the child process."); sleep(2); println("Child again.") end)
sleep(2)
println("This is the parent process again.")
sleep(2)
println("Parent again.")