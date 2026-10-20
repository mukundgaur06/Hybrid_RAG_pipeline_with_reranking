# Principles of Modern Cloud Architecture

## Cloud Service Delivery Models
Modern cloud computing is typically categorized into three fundamental tiers:

1. **Infrastructure as a Service (IaaS)**
   - Delivers raw compute instances, storage volumes, and virtual networks.
   - The cloud consumer manages operating systems, middleware, runtime environments, and applications.
   - Examples: Virtual Machines, Elastic Block Storage, Virtual Private Clouds.

2. **Platform as a Service (PaaS)**
   - Provides managed runtime environments and developer tooling where the underlying operating system, scaling, and patching are abstracted away.
   - The developer focuses purely on application source code and business logic.
   - Examples: Managed container runtimes, Serverless Functions, Managed relational database clusters.

3. **Software as a Service (SaaS)**
   - Fully managed software applications delivered directly over HTTP/HTTPS to end users.
   - The service provider assumes complete responsibility for operations, security patches, high availability, and infrastructure.

## High Availability and Service Level Agreements (SLAs)
Cloud providers guarantee service uptime through formal contracts known as Service Level Agreements. An SLA is quantified by the percentage of uptime delivered during a billing cycle:
- **99.9% Uptime ("Three Nines")**: Permits up to 43.8 minutes of unscheduled downtime per month (8.77 hours per year).
- **99.99% Uptime ("Four Nines")**: Permits up to 4.38 minutes of unscheduled downtime per month (52.6 minutes per year).
- **99.999% Uptime ("Five Nines")**: Highly resilient mission-critical threshold, allowing no more than 26.3 seconds of downtime per month (5.26 minutes per year).

To achieve 99.99% or higher resilience, distributed applications deploy redundant microservice instances across multiple distinct Availability Zones (AZs) backed by independent electrical substations and fiber connectivity.

## Distributed Caching and Latency Budgets
High-throughput systems maintain strict latency budgets (often sub-50 milliseconds for 99th percentile response times). To prevent relational database bottlenecks, engineers insert in-memory caching tiers (such as Redis or Memcached).
- **Cache-Aside Pattern**: The application reads from the cache first; if a cache miss occurs, data is fetched from the primary relational database and asynchronously populated into the cache with an explicit Time-To-Live (TTL).
- **Write-Through Pattern**: Data is written simultaneously to both the cache and the permanent database store, ensuring consistent reads at the cost of higher write latency.
