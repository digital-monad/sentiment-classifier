# MLOps Task

Following are brief docs explaining the parts and design choices behind the implementation.

## Model

The model used is a variant of the BERT architecture called DistilBERT. This was chosen because it is a smaller and faster version of the
full BERT algorithm that maintains a similar level of performance. This model was pretrained to learn useful embeddings for
text sequence classification tasks, and then fine-tuned on the dataset provided to obtain a sntiment classifier that works on the data.
We use ratings as a proxy for sentiment, and train and evaluate the classifier accordingly.

## Inference

Inference is realised as an API endpoint that accepts a POST request with a JSON payload containing the text to classify and some
metadata such as prodict id and timestamp. It is designed for low latency single predictions, and is hosted inside a Docker container
so can be easily deployed to a cloud service, and run via kubernetes etc. Data validation and models are provided viat the `pydantic`
library, and the FastAPI framework is used to handle the API requests. The model is loaded from a serialised format into memory
on application startup.

A number of potential errors with the api schema, both request and response, are handled by `pydantic`'s validation, as defined in
`src/schemas/rest.py`. This ensures that such errors are caught and an appropriate error is thrown.

## Monitoring

Application monitoring is provided via Prometheus and Apitally. The api automatically exposes metrics via the `/metrics` endpoint
which Prometheus scrapes, and is also visualised in Apitally, offering visibility into performance and usage metrics. We observe a p95
latency of 130ms after performing a limited synchronous load test.

Time permitting, there would also be model monitoring in place to catalog and detect drift, performance etc. This would be achieved
by writing the predictions and inputs to a database (as a fastapi background task which is analagous to an async task queue to prevent
blocking). We could then run some sort of batch process to compute statistics like data drift using embeddings, classification metrics
and distribution of predictions over time etc using libraries such as `evidently`.

## Productionisation & Scalability

As mentioned, by hosting the API and monitoring tools inside docker containers (here connected by docker-compose), we ensure smooth transition
to cloud services e.g. ECS, EKS. This means it can also scale vertically (by increasing the resources available to each container) and horizontally
(by increasing the number of replicas) to manage load as the application grows, with the api, model and monitor growing independently of one another.

## Testing

Testing mainly utilises `fastapi`'s Starlette-based test client framework and `pytest`, testing a range of cases where the api is expected to return
a positive or negative sentiment, or throw a validation error.

## Structure

The `src` directory contains all the code for the training and api parts. It is split up into modules to allow for better extensibility and less coupling
beetween the parts.

## Performance

A few considerations were made to optimise performance. Firstly, as mentioned, we use DistilBERT instead of the full BERT model to keep it lightweight and fast.
We also use the FastAPI framework with a minimal amount of middleware (only enough for monitoring) ot ensure minimal latency. Since the `predict` endpoint is caught
up doing cpu intensive work, we make that function synchronous, so that it can be run in a threadpool, which is more efficient for this than async since there is no I/O.
This is in contrast to the health check, which shouldn't block the main thread so is async. Finally, if predictions / data were to be wrtten to stream or db, this would
go inside a background task in fastapi so it is run async and does not increase latency by blocking the main thread as the predict function runs.
