import chess, chess.engine, asyncio
from time import time

async def get_move(b_queue: asyncio.Queue, m_queue: asyncio.Queue):
    _, engine = await chess.engine.popen_uci("./stockfish_exec")
    while True:
        try:
            board = await b_queue.get()
            result = await engine.play(board, chess.engine.Limit(time = 0.1))
            m_queue.put_nowait(result)
            b_queue.task_done()
        except asyncio.CancelledError:
            await engine.quit()
            break
        except Exception as err:
            m_queue.put_nowait(err)
            b_queue.task_done()

async def runner(boards, workers):
    b_queue = asyncio.Queue()
    m_queue = asyncio.Queue()
    for _ in range(boards):
        b_queue.put_nowait(chess.Board())
    
    tasks = [asyncio.create_task(get_move(b_queue, m_queue)) for _ in range(workers)]
    await b_queue.join()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions = True)

    moves = []

    while m_queue.qsize() > 0:
        moves.append(m_queue.get_nowait())

    return moves

loop = asyncio.new_event_loop()

for workers in range(2, 50, 2):
    for boards in range(1, 202, 100):
        s = time()
        moves = loop.run_until_complete(runner(boards, workers))
        print(boards, "/", workers, "->", time() - s, "s")