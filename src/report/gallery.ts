/** Saved reports, kept in the browser's IndexedDB. Nothing leaves the device. */
export interface SavedReport {
  id: string;
  name: string;
  savedAt: number;
  cd: number;
  grade: string;
  thumbnail: string;
  html: string;
}

const DB_NAME = 'wind-tunnel';
const STORE = 'reports';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Could not open the report gallery.'));
  });
}

function run<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const request = action(tx.objectStore(STORE));
        tx.oncomplete = () => {
          db.close();
          resolve(request.result);
        };
        tx.onerror = () => {
          db.close();
          reject(tx.error ?? new Error('The report gallery could not be updated.'));
        };
      }),
  );
}

export function saveReport(report: SavedReport): Promise<IDBValidKey> {
  return run('readwrite', (s) => s.put(report));
}

export async function listReports(): Promise<SavedReport[]> {
  const all = await run<SavedReport[]>('readonly', (s) => s.getAll());
  return all.sort((a, b) => b.savedAt - a.savedAt);
}

export function deleteReport(id: string): Promise<undefined> {
  return run('readwrite', (s) => s.delete(id));
}
