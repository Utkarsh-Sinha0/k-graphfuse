# Security policy

Version 0.1.x is a research release without a security support SLA. Do not submit credentials, private customer graphs or identifying system logs in public issues.

For a sensitive issue, use GitHub's private vulnerability reporting control when available. Ordinary correctness and performance bugs belong in issues with a minimal nonprivate reproducer.

Package installation does not execute benchmark bootstraps. Historical benchmark fixtures download external artifacts and must run in a disposable workspace. Read-only plan mappings are not a hostile-code isolation boundary. CUDA executors must not be shared concurrently.
