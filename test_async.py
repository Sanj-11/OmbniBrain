import asyncio
import time
import httpx

URL = "http://127.0.0.1:8000/query/"
TOKEN = "YOUR_TOKEN_HERE"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

payloads = [
    {
        "text": "show the first 2 rows",
        "dataset_id": 3
    },
    {
        "text": "show the first 5 rows",
        "dataset_id": 3
    },
    {
        "text": "show the dataset summary",
        "dataset_id": 3
    },
]


async def send_query(client, payload, number):
    start = time.perf_counter()

    response = await client.post(
        URL,
        headers=headers,
        json=payload
    )

    elapsed = time.perf_counter() - start

    print(f"Request {number}: {response.status_code} - {elapsed:.3f}s")
    print(response.text)
    print("-" * 50)

    return response


async def main():
    start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            send_query(client, payloads[0], 1),
            send_query(client, payloads[1], 2),
            send_query(client, payloads[2], 3),
        )

    total = time.perf_counter() - start

    print(f"Total execution time: {total:.3f}s")


asyncio.run(main())