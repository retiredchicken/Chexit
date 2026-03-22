import { describe, it, expect } from 'vitest';
import { doc, setDoc, getDoc, deleteDoc, serverTimestamp } from 'firebase/firestore';
import { db } from './firebase';

/**
 * Live Firestore check against your Firebase project.
 * Opt-in so CI and `npm test` never call the network by default.
 *
 * 1. Copy `.env.example` → `.env` and set your `VITE_FIREBASE_*` values (or rely on firebase.ts fallbacks).
 * 2. Set `VITE_RUN_FIREBASE_INTEGRATION=1` in `.env` (or export for one run).
 * 3. Firestore rules must allow create/read/delete on `__vitest_connectivity/{docId}`.
 *
 * Storage uploads are not asserted here: the Firebase *web* Storage client is unreliable in Node
 * (and hits CORS in jsdom/happy-dom). Confirm file upload with `npm run dev` and the Hero control.
 *
 * Run: `npm run test:integration`
 */
const enabled = import.meta.env.VITE_RUN_FIREBASE_INTEGRATION === '1';

describe.skipIf(!enabled)('Firebase integration (live project)', () => {
  it(
    'Firestore: write, read, and delete a scratch document',
    async () => {
      const id = crypto.randomUUID();
      const refDoc = doc(db, '__vitest_connectivity', id);
      await setDoc(refDoc, { testedAt: serverTimestamp(), source: 'vitest' });
      const snap = await getDoc(refDoc);
      expect(snap.exists()).toBe(true);
      await deleteDoc(refDoc);
      const after = await getDoc(refDoc);
      expect(after.exists()).toBe(false);
    },
    60_000,
  );
});
